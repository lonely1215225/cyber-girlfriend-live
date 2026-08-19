"""Queued @小麻 room replies using the idle speech-to-speech pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx
import websockets


logger = logging.getLogger("s2s.mention_reply")
MENTION_RE = re.compile(r"@小麻(?:\s*[：:,，]?\s*)", re.IGNORECASE)


@dataclass(slots=True)
class MentionRequest:
    message_id: str
    speaker: str
    text: str
    prompt: str


def parse_mention(message: dict[str, Any]) -> MentionRequest | None:
    text = str(message.get("text") or "").strip()
    match = MENTION_RE.search(text)
    if not match:
        return None
    prompt = text[match.end() :].strip()
    if not prompt:
        prompt = "有人在直播间叫你，请自然地回应对方，并问一句容易接话的小问题。"
    return MentionRequest(
        message_id=str(message.get("id") or ""),
        speaker=str(message.get("speaker") or "观众"),
        text=text,
        prompt=prompt,
    )


class MentionReplyWorker:
    def __init__(self, room, mcp_gateway, ws_url: str, avatar_url: str = "", max_queue: int = 30):
        self.room = room
        self.mcp_gateway = mcp_gateway
        self.ws_url = ws_url
        self.avatar_url = avatar_url.rstrip("/")
        self.max_queue = max(1, max_queue)
        self.pending: deque[MentionRequest] = deque()
        self._wake = asyncio.Event()
        self._runner: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None

    def start(self) -> None:
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._run())

    async def stop(self) -> None:
        await self.interrupt()
        if self._runner:
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner
            self._runner = None

    def enqueue(self, message: dict[str, Any]) -> int | None:
        request = parse_mention(message)
        if request is None:
            return None
        if len(self.pending) >= self.max_queue:
            self.pending.popleft()
        self.pending.append(request)
        self._wake.set()
        return len(self.pending)

    def notify(self) -> None:
        self._wake.set()

    async def interrupt(self) -> None:
        task = self._response_task
        was_active = bool(task and not task.done())
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if was_active and self.avatar_url:
            with contextlib.suppress(httpx.HTTPError):
                async with httpx.AsyncClient(timeout=2.0) as client:
                    await client.post(f"{self.avatar_url}/interrupt")

    async def _run(self) -> None:
        while True:
            if not self.pending:
                self._wake.clear()
                await self._wake.wait()
                continue
            if not await self.room.can_bot_reply():
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                continue
            request = self.pending.popleft()
            self._response_task = asyncio.create_task(self._respond(request))
            try:
                await self._response_task
            except asyncio.CancelledError:
                # A live caller preempted this reply. Preserve FIFO and retry
                # after the room becomes idle again.
                self.pending.appendleft(request)
            except Exception as exc:  # noqa: BLE001
                logger.warning("@小麻 reply failed for %s: %s", request.message_id, exc)
            finally:
                self._response_task = None

    async def _respond(self, request: MentionRequest) -> None:
        tools = await self.mcp_gateway.list_tools() if self.mcp_gateway.enabled else []
        public_tools = [
            {key: tool[key] for key in ("type", "name", "description", "parameters")}
            for tool in tools
        ]
        instructions = (
            "你叫小麻，是直播间里的数字人。说中文口语，一到两句，自然、有情绪，不用Markdown。"
            "你正在回复公开评论，不要假装对方正在语音连线。"
            "如果问题涉及实时价格或最新新闻，必须立即调用提供的工具，拿到结果后直接回答。"
        )
        user_text = f"直播间观众“{request.speaker}”评论：{request.prompt}"
        transcript = ""
        pending_calls: list[dict[str, str]] = []
        tool_rounds = 0
        async with websockets.connect(self.ws_url, max_size=None, ping_interval=20, ping_timeout=20) as ws:
            while True:
                event = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                if event.get("type") == "session.created":
                    break
            session: dict[str, Any] = {
                "type": "realtime",
                "instructions": instructions,
                "audio": {"output": {"voice": "Vivian"}},
            }
            if public_tools:
                session.update(tools=public_tools, tool_choice="auto")
            await ws.send(json.dumps({"type": "session.update", "session": session}, ensure_ascii=False))
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": user_text}],
                        },
                    },
                    ensure_ascii=False,
                )
            )
            await ws.send(json.dumps({"type": "response.create", "response": {}}))

            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=90)
                event = json.loads(raw)
                event_type = str(event.get("type") or "")
                if event_type in {"response.audio_transcript.delta", "response.output_audio_transcript.delta"}:
                    transcript += str(event.get("delta") or "")
                elif event_type in {"response.audio_transcript.done", "response.output_audio_transcript.done"}:
                    transcript = str(event.get("transcript") or transcript)
                elif event_type == "response.function_call_arguments.done":
                    pending_calls.append(
                        {
                            "name": str(event.get("name") or ""),
                            "arguments": str(event.get("arguments") or "{}"),
                            "call_id": str(event.get("call_id") or ""),
                        }
                    )
                elif event_type == "response.done":
                    if pending_calls and tool_rounds < 3:
                        tool_rounds += 1
                        calls, pending_calls = pending_calls, []
                        for call in calls:
                            try:
                                arguments = json.loads(call["arguments"])
                                output = await self.mcp_gateway.call(call["name"], arguments)
                            except Exception as exc:  # noqa: BLE001
                                output = f"工具调用失败：{exc}"
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": call["call_id"],
                                            "output": output,
                                        },
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        await ws.send(json.dumps({"type": "response.create", "response": {}}))
                        transcript = ""
                        continue
                    await self.room.publish_bot_reply(
                        message_id=request.message_id,
                        text=transcript,
                        reply_to={
                            "id": request.message_id,
                            "speaker": request.speaker,
                            "text": request.text,
                        },
                    )
                    return
