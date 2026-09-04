"""Server-owned live-mic lookup: prefetch or pin news before the host answers."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from avatar_profiles import (
    DEFAULT_PERSONA_PROMPT,
    ROLE_IDENTITY_POLICY,
    ROLE_OUTPUT_POLICY,
)
from dialogue_intent import (
    COMPANION_POLICY,
    LOOKED_UP_EVIDENCE_POLICY,
    LOOKUP_FAIL_LINE,
    PINNED_TOPIC_POLICY,
    READ_EXACT_INSTRUCTIONS,
    SIMPLE_CHAT_POLICY,
    SPOKEN_CHINESE_POLICY,
    decide_voice_turn,
    lookup_wait_line,
    viewer_utterance,
)


logger = logging.getLogger("s2s.voice_lookup")
SendFn = Callable[[dict[str, Any]], Awaitable[None]]
FetchFn = Callable[[str], Awaitable[str]]
_RESPONSE_PREFIXES = ("response.",)
_RESPONSE_TYPES = frozenset({
    "response.created",
    "response.done",
    "response.cancelled",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.audio.delta",
    "response.output_audio.delta",
    "response.audio.done",
    "response.output_audio.done",
    "response.audio_transcript.delta",
    "response.output_audio_transcript.delta",
    "response.audio_transcript.done",
    "response.output_audio_transcript.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
})


def compose_voice_instructions(
    persona_prompt: str,
    display_name: str,
    *,
    personal_memory: str = "",
    active_news: str = "",
    evidence: bool = False,
    pinned: bool = False,
    companion: bool = False,
) -> str:
    """Persona + one turn policy. Search tools are never offered here."""
    instructions = str(persona_prompt or DEFAULT_PERSONA_PROMPT).strip()
    identity = f"当前正在与你连线的观众名字是“{display_name}”。请自然地用这个名字称呼对方。"
    if evidence:
        policy = LOOKED_UP_EVIDENCE_POLICY + SPOKEN_CHINESE_POLICY
    elif pinned:
        policy = PINNED_TOPIC_POLICY + SPOKEN_CHINESE_POLICY
    elif companion:
        policy = COMPANION_POLICY
    else:
        policy = (
            "正在连线，不要假装在回评论。"
            f"{SIMPLE_CHAT_POLICY}{SPOKEN_CHINESE_POLICY}"
        )
    additions = [
        item for item in (ROLE_IDENTITY_POLICY, ROLE_OUTPUT_POLICY, identity, policy)
        if item and item not in instructions
    ]
    if personal_memory:
        additions.append(
            "以下记忆仅属于当前连线者，只在相关时自然使用，不复述、不与其他用户混用。"
            "\n【当前用户个人记忆】\n"
            f"{personal_memory}"
        )
    if active_news:
        additions.append(
            "下面是直播间刚播报的公共话题，可用于承接讨论。"
            "只有对方说‘这个、刚才那条、它、为什么、后来呢’或明确提到相关主体时才使用；"
            "无关问题必须忽略。\n"
            f"{active_news}"
        )
    return "\n".join([instructions, *additions]).strip()


def strip_live_tools(session_data: dict[str, Any]) -> dict[str, Any]:
    """Live voice never exposes search tools; the server looks facts up."""
    session_data["tools"] = []
    session_data["tool_choice"] = "none"
    return session_data


def cancel_response_message() -> dict[str, Any]:
    return {"type": "response.cancel"}


def read_exact_messages(phrase: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "session.update",
            "session": strip_live_tools({
                "type": "realtime",
                "instructions": READ_EXACT_INSTRUCTIONS,
                "audio": {"output": {"voice": "active_profile"}},
            }),
        },
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"逐字朗读：{phrase}"}],
            },
        },
        {
            "type": "response.create",
            "response": {"metadata": {"client_purpose": "tool_progress"}},
        },
    ]


def evidence_turn_messages(
    instructions: str,
    evidence: str,
    display_name: str,
    utterance: str,
    *,
    pinned: bool = False,
) -> list[dict[str, Any]]:
    label = "【刚才播过的话题】" if pinned else "【已查到的资料】"
    user_text = (
        f"{label}\n{evidence}\n\n"
        f"【当前这句，据此回答】\n连线观众“{display_name}”说：{utterance}"
    )
    return [
        {
            "type": "session.update",
            "session": strip_live_tools({
                "type": "realtime",
                "instructions": instructions,
                "audio": {"output": {"voice": "active_profile"}},
            }),
        },
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            },
        },
        {"type": "response.create", "response": {}},
    ]


def chat_retry_messages(instructions: str, display_name: str, utterance: str) -> list[dict[str, Any]]:
    user_text = f"连线观众“{display_name}”说：{utterance}"
    return [
        {
            "type": "session.update",
            "session": strip_live_tools({
                "type": "realtime",
                "instructions": instructions,
                "audio": {"output": {"voice": "active_profile"}},
            }),
        },
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            },
        },
        {"type": "response.create", "response": {}},
    ]


def is_response_event(event_type: str) -> bool:
    return event_type in _RESPONSE_TYPES or event_type.startswith(_RESPONSE_PREFIXES)


class VoiceLookupGate:
    """Hold the auto-answer while the server fetches or pins evidence."""

    def __init__(self) -> None:
        self.holding = False
        self.accept_responses = False
        self.barge_in = False
        self.expect_cancel_done = False
        self.expect_progress_done = False
        self.cancel_acked = asyncio.Event()
        self.progress_done = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.utterance = ""
        self.generation = 0

    def begin(self, utterance: str) -> int:
        self.generation += 1
        self.holding = True
        self.accept_responses = False
        self.barge_in = False
        self.expect_cancel_done = True
        self.expect_progress_done = False
        self.cancel_acked = asyncio.Event()
        self.progress_done = asyncio.Event()
        self.utterance = utterance
        return self.generation

    def allow_own_responses(self) -> None:
        self.accept_responses = True

    def release(self) -> None:
        self.holding = False
        self.accept_responses = False
        self.expect_cancel_done = False
        self.expect_progress_done = False
        self.utterance = ""

    def interrupt(self) -> None:
        self.barge_in = True
        task = self.task
        self.task = None
        if task is not None and not task.done():
            task.cancel()
        self.release()

    def should_drop_upstream(self, event_type: str) -> bool:
        if not self.holding or self.accept_responses:
            return False
        return is_response_event(event_type)

    def should_hold_browser_create(self) -> bool:
        return self.holding

    def observe(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type != "response.done":
            return
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        metadata = response.get("metadata") if isinstance(response.get("metadata"), dict) else {}
        if self.expect_progress_done and metadata.get("client_purpose") == "tool_progress":
            self.progress_done.set()
            self.expect_progress_done = False
            return
        if self.expect_cancel_done:
            self.cancel_acked.set()
            self.expect_cancel_done = False


async def _send_all(send: SendFn, messages: list[dict[str, Any]]) -> None:
    for message in messages:
        await send(message)


async def run_voice_lookup(
    *,
    utterance: str,
    display_name: str,
    persona_prompt: str,
    personal_memory: str,
    companion: bool,
    kind: str,
    send: SendFn,
    prefetch: FetchFn,
    pin_news: FetchFn,
    gate: VoiceLookupGate,
) -> None:
    """Cancel the auto-answer, attach evidence, then create the real turn."""
    spoken = viewer_utterance(utterance)
    generation = gate.generation
    try:
        await send(cancel_response_message())
        try:
            await asyncio.wait_for(gate.cancel_acked.wait(), timeout=0.25)
        except asyncio.TimeoutError:
            pass
        if gate.barge_in:
            return

        evidence = ""
        if kind == "prefetch":
            prefetch_task = asyncio.create_task(prefetch(spoken))
            await asyncio.sleep(0)
            if not prefetch_task.done() and not gate.barge_in:
                wait_line = lookup_wait_line(spoken)
                gate.expect_progress_done = True
                gate.progress_done = asyncio.Event()
                await _send_all(send, read_exact_messages(wait_line))
                gate.allow_own_responses()
                try:
                    await asyncio.wait_for(gate.progress_done.wait(), timeout=12)
                except asyncio.TimeoutError:
                    logger.warning("voice wait line did not finish")
            try:
                evidence = str(await prefetch_task or "").strip()
            except asyncio.CancelledError:
                prefetch_task.cancel()
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("voice lookup prefetch failed: %s", exc)
                evidence = ""
            if gate.barge_in:
                return
            if not evidence:
                gate.allow_own_responses()
                await _send_all(send, read_exact_messages(LOOKUP_FAIL_LINE))
                return
            instructions = compose_voice_instructions(
                persona_prompt, display_name,
                personal_memory=personal_memory, evidence=True, companion=companion,
            )
            await _send_all(
                send,
                evidence_turn_messages(instructions, evidence, display_name, spoken),
            )
            return

        evidence = str(await pin_news(spoken) or "").strip()
        if gate.barge_in:
            return
        if evidence:
            instructions = compose_voice_instructions(
                persona_prompt, display_name,
                personal_memory=personal_memory, pinned=True, companion=companion,
            )
            await _send_all(
                send,
                evidence_turn_messages(
                    instructions, evidence, display_name, spoken, pinned=True,
                ),
            )
            return
        instructions = compose_voice_instructions(
            persona_prompt, display_name,
            personal_memory=personal_memory, companion=companion,
        )
        await _send_all(send, chat_retry_messages(instructions, display_name, spoken))
    finally:
        if not gate.barge_in and gate.generation == generation:
            gate.allow_own_responses()
            gate.holding = False


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
