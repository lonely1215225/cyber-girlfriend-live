"""Queued @小麻 room replies using the idle speech-to-speech pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import websockets

from avatar_profiles import DEFAULT_PERSONA_PROMPT, ROLE_IDENTITY_POLICY, ROLE_OUTPUT_POLICY
from dialogue_intent import (
    LOOKED_UP_EVIDENCE_POLICY,
    SIMPLE_CHAT_POLICY,
    SPOKEN_CHINESE_POLICY,
    looks_like_english_answer,
    looks_like_search_filler,
    needs_web_search,
    wants_news_context,
)


logger = logging.getLogger("s2s.mention_reply")
ROOM_TIMEZONE = os.environ.get("AGENT_TIMEZONE", "Asia/Shanghai").strip() or "Asia/Shanghai"
MENTION_RE = re.compile(r"@小麻(?:\s*[：:,，]?\s*)", re.IGNORECASE)
DEFERRED_ANSWER_RE = re.compile(
    r"(?:我|这就|马上)?去(?:帮你)?查|我查查|我看看|稍等|等一下|马上回来|查完告诉你",
    re.IGNORECASE,
)
SESSION_BUSY_RE = re.compile(r"all session slots are in use|session slots.*(?:busy|full)", re.I)
MAX_SESSION_BUSY_RETRIES = 8
DIALOGUE_TOOLS_ENABLED = os.environ.get("DIALOGUE_TOOLS_ENABLED", "0") == "1"


def mention_failure_reply(exc: BaseException) -> str:
    """Speak a viewer-facing fallback. Busy is not the same as a blank answer."""
    if SESSION_BUSY_RE.search(str(exc)):
        return "刚才说话的通道还在忙，我没接上这一句。你再说一次我就回你。"
    return "这一句我没答上来。你换个说法再叫我一次就好。"


def _public_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    return {key: tool[key] for key in ("type", "name", "description", "parameters") if key in tool}


class DuplicateReplyDetected(RuntimeError):
    """Raised before publication when a new turn starts replaying the last answer."""


@dataclass(slots=True)
class MentionRequest:
    message_id: str
    participant_id: str
    speaker: str
    text: str
    prompt: str
    proactive: bool = False
    welcome: bool = False
    session_busy_retries: int = 0
    created_at: float = 0.0
    delivery_started: bool = False
    final_delivered: bool = False
    delivered_text: str = ""
    superseded: bool = False
    duplicate_retries: int = 0
    avoid_reply: str = ""


def looks_like_deferred_answer(text: str) -> bool:
    value = str(text or "").strip()
    return bool(DEFERRED_ANSWER_RE.search(value)) or looks_like_search_filler(value)


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
        participant_id=str(message.get("participant_id") or ""),
        speaker=str(message.get("speaker") or "观众"),
        text=text,
        prompt=prompt,
        created_at=float(message.get("created_at") or time.time()),
    )


class MentionReplyWorker:
    def __init__(self, room, mcp_gateway, ws_url: str, avatar_url: str = "", max_queue: int = 30,
                 persona_provider=None):
        self.room = room
        self.mcp_gateway = mcp_gateway
        self.ws_url = ws_url
        self.avatar_url = avatar_url.rstrip("/")
        self.max_queue = max(1, max_queue)
        self.persona_provider = persona_provider
        self.pending: deque[MentionRequest] = deque()
        self._wake = asyncio.Event()
        self._runner: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None
        self._active_request: MentionRequest | None = None
        self._speech_sequence = 0

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
        if self._active_request and self._active_request.message_id == request.message_id:
            return 0
        for index, pending in enumerate(self.pending):
            if pending.message_id == request.message_id:
                return index + 1
        # A direct viewer message makes their unplayed welcome stale.  It also
        # supersedes older unplayed questions from the same viewer and any idle
        # broadcast waiting in front of it.  Otherwise a busy room appears to
        # answer one new comment three or four times in succession.
        kept: deque[MentionRequest] = deque()
        stale: list[MentionRequest] = []
        for pending in self.pending:
            should_drop = pending.proactive or (
                pending.participant_id == request.participant_id
                and (pending.welcome or not pending.delivery_started)
            )
            if should_drop:
                pending.superseded = True
                stale.append(pending)
            else:
                kept.append(pending)
        self.pending = kept
        for old in stale:
            self._schedule_superseded(old)

        active = self._active_request
        if active is not None and (
            active.proactive
            or (
                active.participant_id == request.participant_id
                and not active.delivery_started
            )
        ):
            active.superseded = True
            task = self._response_task
            if task is not None and not task.done():
                task.cancel()
                self._schedule_avatar_interrupt()
        if len(self.pending) >= self.max_queue:
            self.pending.popleft()
        insert_at = next(
            (index for index, pending in enumerate(self.pending) if pending.welcome or pending.proactive),
            len(self.pending),
        )
        self.pending.insert(insert_at, request)
        self._wake.set()
        return insert_at + 1

    def _schedule_superseded(self, request: MentionRequest) -> None:
        if request.proactive or request.welcome:
            return
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(self._finalize_superseded(request))

    def _schedule_avatar_interrupt(self) -> None:
        if not self.avatar_url:
            return

        async def stop_playback() -> None:
            with contextlib.suppress(httpx.HTTPError):
                async with httpx.AsyncClient(timeout=2.0) as client:
                    await client.post(f"{self.avatar_url}/interrupt")

        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(stop_playback())

    async def publish_queued(self, message: dict[str, Any], position: int) -> None:
        """Expose queue acknowledgement immediately instead of waiting silently."""
        request = parse_mention(message)
        if request is None:
            return
        busy = not await self.room.can_bot_reply()
        status = (
            f"当前正在连线，已进入回复队列（第 {max(1, position)} 位）"
            if busy else "已收到，正在准备回复"
        )
        await self.room.publish_agent_job({
            "id": self._agent_job_id(request), "message_id": request.message_id,
            "participant_id": request.participant_id, "speaker": request.speaker,
            "prompt": request.prompt, "phase": "queued", "status_text": status,
            "terminal": False, "created_at": time.time(), "event_only": True,
        }, reply_to=self._reply_quote(request))

    def enqueue_proactive(self, prompt: str) -> bool:
        """Queue one unquoted room-wide topic without duplicating idle jobs."""
        if any(request.proactive for request in self.pending):
            return False
        self.pending.append(
            MentionRequest(
                message_id=f"proactive:{int(time.time() * 1000)}",
                participant_id="",
                speaker="小雅",
                text="",
                prompt=prompt.strip(),
                proactive=True,
            )
        )
        self._wake.set()
        return True

    def enqueue_welcome(self, *, participant_id: str, speaker: str) -> bool:
        """Queue one LLM-generated arrival welcome ahead of idle broadcasts."""
        message_id = f"welcome:{participant_id}:{int(time.time() * 1000)}"
        if any(request.welcome and request.participant_id == participant_id for request in self.pending):
            return False
        if len(self.pending) >= self.max_queue:
            disposable = next(
                (index for index, request in enumerate(self.pending) if request.proactive or request.welcome),
                None,
            )
            if disposable is None:
                return False
            del self.pending[disposable]
        request = MentionRequest(
            message_id=message_id,
            participant_id=participant_id,
            speaker=str(speaker or "新来的朋友")[:24],
            text="",
            prompt="生成一条入场欢迎词",
            welcome=True,
        )
        # Priority: live call > @ reply > arrival welcome > idle news.
        insert_at = next(
            (index for index, pending in enumerate(self.pending) if pending.proactive),
            len(self.pending),
        )
        self.pending.insert(insert_at, request)
        self._wake.set()
        return True

    def notify(self) -> None:
        self._wake.set()

    def drop_welcomes(self, participant_id: str = "") -> int:
        """Discard arrival greetings made stale by a live call grant."""
        kept: deque[MentionRequest] = deque()
        removed = 0
        for request in self.pending:
            if request.welcome and (
                not participant_id or request.participant_id == participant_id
            ):
                removed += 1
                continue
            kept.append(request)
        self.pending = kept
        return removed

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
            if request.proactive and not await self.room.can_start_proactive():
                logger.info("dropped idle news because the room is empty or busy")
                continue
            self._active_request = request
            self._response_task = asyncio.create_task(self._respond(request))
            try:
                await self._response_task
            except asyncio.CancelledError:
                if request.superseded:
                    await self._finalize_superseded(request)
                # A live caller preempted this reply. Retry only work which has
                # not emitted anything; replaying an already visible/audible
                # response after the call would duplicate it from the start.
                elif request.delivery_started:
                    await self._finalize_interrupted_delivery(request)
                elif not request.welcome:
                    self.pending.appendleft(request)
                    await self._reset_agent_job(request)
            except DuplicateReplyDetected as exc:
                logger.warning(
                    "blocked repeated reply for %s (retry=%d): %s",
                    request.message_id,
                    request.duplicate_retries,
                    exc,
                )
                self._schedule_avatar_interrupt()
                if request.duplicate_retries < 1 and not request.superseded:
                    request.duplicate_retries += 1
                    request.delivery_started = False
                    request.final_delivered = False
                    request.delivered_text = ""
                    self.pending.appendleft(request)
                    await self._reset_agent_job(request)
                    self._wake.set()
                elif not request.proactive and not request.welcome:
                    await self.room.publish_agent_job({
                        "id": self._agent_job_id(request), "message_id": request.message_id,
                        "participant_id": request.participant_id, "speaker": request.speaker,
                        "prompt": request.prompt, "phase": "failed",
                        "status_text": "检测到重复回答，本轮已停止", "terminal": True,
                        "error": "duplicate_reply", "event_only": True,
                    }, reply_to=self._reply_quote(request))
            except Exception as exc:  # noqa: BLE001
                logger.warning("@小麻 reply failed for %s: %s", request.message_id, exc)
                # If anything from this task was already visible/audible, do
                # not append a second generic failure answer much later. That
                # stale fallback looks like a reply to the viewer's next
                # comment and compounds the original failure.
                if request.delivery_started:
                    if not request.proactive and not request.welcome:
                        await self.room.publish_agent_job({
                            "id": self._agent_job_id(request), "message_id": request.message_id,
                            "participant_id": request.participant_id, "speaker": request.speaker,
                            "prompt": request.prompt, "phase": "failed",
                            "status_text": "本轮回复未完整完成", "terminal": True,
                            "error": str(exc), "event_only": True,
                        }, reply_to=self._reply_quote(request))
                    continue
                if (
                    SESSION_BUSY_RE.search(str(exc))
                    and request.session_busy_retries < MAX_SESSION_BUSY_RETRIES
                ):
                    request.session_busy_retries += 1
                    self.pending.appendleft(request)
                    await asyncio.sleep(min(5.0, 1.0 * (2 ** (request.session_busy_retries - 1))))
                    self._wake.set()
                    continue
                if not request.proactive and not request.welcome:
                    await self.room.publish_agent_job({
                        "id": self._agent_job_id(request), "message_id": request.message_id,
                        "participant_id": request.participant_id, "speaker": request.speaker,
                        "prompt": request.prompt, "phase": "failed", "status_text": "本轮处理失败",
                        "terminal": True, "error": str(exc), "event_only": True,
                    }, reply_to=self._reply_quote(request))
                    await self.room.publish_bot_reply(
                        message_id=request.message_id,
                        text=mention_failure_reply(exc),
                        reply_to=self._reply_quote(request),
                    )
            finally:
                self._response_task = None
                self._active_request = None

    async def restore_jobs(self) -> None:
        """Resume persisted jobs and very recent mentions accepted before a restart."""
        store = getattr(self.room, "store", None)
        if store is None:
            return
        jobs = await store.load_agent_jobs(recoverable_only=True, limit=self.max_queue)
        known_jobs = await store.load_agent_jobs(limit=100)
        known_message_ids = {str(job.get("message_id") or "") for job in known_jobs}
        queued_ids = {request.message_id for request in self.pending}
        for job in jobs:
            if job["message_id"] in queued_ids:
                continue
            self.pending.append(MentionRequest(
                message_id=job["message_id"], participant_id=job.get("participant_id") or "",
                speaker=job.get("speaker") or "观众", text=f"@小麻 {job.get('prompt') or ''}",
                prompt=job.get("prompt") or "", proactive=False,
            ))
            queued_ids.add(job["message_id"])

        # A comment is stored before it is enqueued. If the process stops in
        # that tiny window—or while a call owns the worker—the comment must not
        # disappear. Recover only a short window and exclude every known job,
        # so old completed mentions are never replayed.
        cutoff = time.time() - 15 * 60
        recent = await store.load_recent_messages(80)
        recovered = 0
        for message in recent:
            message_id = str(message.get("id") or "")
            if (
                float(message.get("created_at") or 0) < cutoff
                or message_id in known_message_ids
                or message_id in queued_ids
            ):
                continue
            position = self.enqueue(message)
            if position is None:
                continue
            queued_ids.add(message_id)
            await self.publish_queued(message, position)
            recovered += 1
        if jobs or recovered:
            self._wake.set()

    @staticmethod
    def _agent_job_id(request: MentionRequest) -> str:
        digest = hashlib.sha256(f"default:{request.message_id}".encode()).hexdigest()[:24]
        return f"aj_{digest}"

    async def _reset_agent_job(self, request: MentionRequest) -> None:
        if request.proactive or request.welcome:
            return
        await self.room.publish_agent_job({
            "id": self._agent_job_id(request), "message_id": request.message_id,
            "participant_id": request.participant_id, "speaker": request.speaker,
            "prompt": request.prompt, "phase": "queued", "status_text": "连线结束后继续帮你查",
            "terminal": False, "event_only": True,
        }, reply_to=self._reply_quote(request))

    async def _finalize_superseded(self, request: MentionRequest) -> None:
        """Close stale work without publishing another chat reply."""
        if request.proactive or request.welcome:
            return
        await self.room.publish_agent_job({
            "id": self._agent_job_id(request), "message_id": request.message_id,
            "participant_id": request.participant_id, "speaker": request.speaker,
            "prompt": request.prompt, "phase": "cancelled",
            "status_text": "已由你的新消息替代", "terminal": True,
            "error": "superseded_by_new_message", "event_only": True,
        }, reply_to=self._reply_quote(request))

    async def _finalize_interrupted_delivery(self, request: MentionRequest) -> None:
        """Never replay a room reply which the audience has already heard or seen."""
        if request.proactive or request.welcome:
            return
        completed = request.final_delivered and bool(request.delivered_text.strip())
        await self.room.publish_agent_job({
            "id": self._agent_job_id(request), "message_id": request.message_id,
            "participant_id": request.participant_id, "speaker": request.speaker,
            "prompt": request.prompt,
            "phase": "completed" if completed else "cancelled",
            "status_text": (
                request.delivered_text if completed else "回复已被连线打断，不再重复播放"
            ),
            "final_text": request.delivered_text if completed else "",
            "terminal": True,
            "error": "" if completed else "preempted_after_delivery",
            "event_only": True,
        }, reply_to=self._reply_quote(request))

    async def _respond(self, request: MentionRequest) -> None:
        self._speech_sequence += 1
        speech_base_id = f"{request.message_id}:speech:{self._speech_sequence}"
        transcript = ""
        completed_transcript = ""
        segment_delta = ""
        speech_round = 0
        round_finalized = False
        last_partial_at = 0.0
        reply_quote = self._reply_quote(request)
        job = None if request.proactive or request.welcome else {
            "id": self._agent_job_id(request), "message_id": request.message_id,
            "participant_id": request.participant_id, "speaker": request.speaker,
            "prompt": request.prompt, "phase": "responding", "status_text": "正在回复",
            "terminal": False, "created_at": time.time(), "event_only": True,
        }
        if job is not None:
            await self.room.publish_agent_job(job, reply_to=reply_quote)

        previous_reply = ""
        previous_reply_provider = getattr(self.room, "latest_assistant_reply", None)
        if callable(previous_reply_provider) and not request.proactive and not request.welcome:
            previous_reply = await previous_reply_provider(
                request.participant_id, exclude_reply_to_id=request.message_id
            )

        def normalized_for_duplicate(value: str) -> str:
            return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", str(value or "").lower())

        async def publish_spoken(*, partial: bool, interrupted: bool = False) -> None:
            nonlocal last_partial_at, round_finalized, transcript, completed_transcript, segment_delta
            if not transcript.strip():
                return
            if looks_like_english_answer(transcript) and not request.proactive and not request.welcome:
                return
            if (
                not request.proactive
                and not request.welcome
                and looks_like_deferred_answer(transcript)
                and len(transcript.strip()) <= 80
            ):
                return
            candidate = normalized_for_duplicate(transcript)
            previous = normalized_for_duplicate(previous_reply)
            # Detect while the first sentence is still inside the playback
            # reservoir. This is an identity/integrity guard, not semantic
            # answer logic: a distinct user turn may not replay the previous
            # answer byte-for-byte just because retrieval ranked it highly.
            if (
                previous
                and len(candidate) >= 18
                and (previous.startswith(candidate) or candidate.startswith(previous))
            ):
                request.avoid_reply = previous_reply[:600]
                transcript = ""
                completed_transcript = ""
                segment_delta = ""
                raise DuplicateReplyDetected("generated text matches previous assistant reply")
            await self.room.publish_bot_reply(
                message_id=f"{speech_base_id}:{speech_round}",
                text=transcript,
                reply_to=reply_quote,
                partial=partial,
                interrupted=interrupted,
                # An arrival welcome is public, but its conversational owner is
                # the person being welcomed. Persist that association so their
                # later turns can naturally refer back to what was just said.
                memory_user_id=request.participant_id if request.welcome else "",
            )
            request.delivery_started = True
            request.delivered_text = transcript.strip()
            if not partial and not interrupted:
                request.final_delivered = True
            last_partial_at = time.monotonic()
            if not partial:
                round_finalized = True

        def join_spoken(left: str, right: str) -> str:
            left, right = left.strip(), right.strip()
            if not left:
                return right
            if not right:
                return left
            separator = " " if left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum() else ""
            return left + separator + right

        # Server-side intent: casual chat is memory-only and has no tools.
        # Only an explicit live-lookup request may see web search.
        wants_search = (
            DIALOGUE_TOOLS_ENABLED
            and not request.proactive
            and not request.welcome
            and needs_web_search(request.prompt)
        )
        prefetch_task: asyncio.Task[str] | None = None
        if wants_search:
            prefetch_fn = getattr(self.mcp_gateway, "prefetch_spoken_evidence", None)
            if callable(prefetch_fn):
                if job is not None:
                    job.update(
                        phase="researching",
                        status_text="正在查最新资讯",
                        terminal=False,
                    )
                    await self.room.publish_agent_job(job, reply_to=reply_quote)
                prefetch_task = asyncio.create_task(prefetch_fn(request.prompt))
        try:
            # News and welcomes use the same live TTS socket as chat: first
            # sentence starts after the short reservoir, not after a
            # whole-clip wait.
            async with websockets.connect(
                self.ws_url, max_size=None, ping_interval=20, ping_timeout=20
            ) as ws:
                while True:
                    event = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                    if event.get("type") == "session.created":
                        break

                prefetched = ""
                if prefetch_task is not None:
                    try:
                        prefetched = str(await prefetch_task or "").strip()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("lookup prefetch failed: %s", exc)
                        prefetched = ""
                web_tool = (
                    self.mcp_gateway.dialogue_web_tool()
                    if wants_search and not prefetched and self.mcp_gateway.enabled
                    else None
                )
                public_tools = [_public_tool_schema(web_tool)] if web_tool else []
                web_tool_name = str(web_tool.get("name") or "") if web_tool else ""
                active_external_tools: dict[str, dict[str, Any]] = {}
                tool_progress = {
                    web_tool_name: str(web_tool.get("progress_text") or "正在查询并核对资料")
                } if web_tool else {}

                persona_prompt = DEFAULT_PERSONA_PROMPT
                if self.persona_provider is not None:
                    try:
                        persona_prompt = await self.persona_provider() or persona_prompt
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("active persona lookup failed: %s", exc)
                instructions = (
                    f"{persona_prompt}\n{ROLE_IDENTITY_POLICY}\n{ROLE_OUTPUT_POLICY}\n"
                    f"当前时间：{datetime.now(ZoneInfo(ROOM_TIMEZONE)).isoformat(timespec='seconds')}"
                    f"（{ROOM_TIMEZONE}）。"
                )
                if request.proactive:
                    instructions += (
                        "现在是无人连线时的直播间主动播报，不要假装在回复某位观众。"
                        "房间里有人在看，只是没人连麦；对着观众说话，不要说直播间没人。"
                        "声音表演要先保证清晰、完整，再根据新闻本身决定轻快、认真、"
                        "惊讶或克制的情绪；严肃事件不撒娇、不发笑。"
                        "必须说完整句子，用句号、问号或感叹号收尾，禁止半句截断。"
                    )
                    user_text = request.prompt
                elif request.welcome:
                    instructions += (
                        "这是直播间入场欢迎。观众刚进来，直接叫名字，一句说完，十八到三十二个字，"
                        "像刚看见对方随口打个招呼。可以甜或俏皮，不要“既然来了那就……”这种书面腔，"
                        "不套模板，不问隐私，不提AI。"
                    )
                    user_text = f"刚进入直播间的观众名字是“{request.speaker}”，现在欢迎对方。"
                else:
                    if prefetched:
                        instructions += (
                            "这是公开评论：不要假装正在连线。"
                            f"{LOOKED_UP_EVIDENCE_POLICY}{SPOKEN_CHINESE_POLICY}"
                        )
                    elif web_tool:
                        instructions += (
                            "这是公开评论：观众要查网上的最新事实。先联网，再用两三句中文口语说结论。"
                            "不要“既然……那就……”这类书面腔，不要假装正在连线。"
                            "不要只说正在查找或核对来源。"
                            f"{SPOKEN_CHINESE_POLICY}"
                        )
                    else:
                        instructions += (
                            "这是公开评论：不要假装正在连线。"
                            f"{SIMPLE_CHAT_POLICY}{SPOKEN_CHINESE_POLICY}"
                        )
                    memory_task = self.room.participant_memory_context(
                        request.participant_id,
                        request.prompt,
                        exclude_message_id=request.message_id,
                    )
                    if wants_news_context(request.prompt):
                        memory, active_topic = await asyncio.gather(
                            memory_task,
                            self.room.active_news_context(request.prompt),
                        )
                    else:
                        memory = await memory_task
                        active_topic = ""
                    context_sections: list[str] = []
                    if memory:
                        instructions += (
                            "附带记忆仅属于当前观众，只能辅助理解指代；历史中的数字人回答不是本轮答案，"
                            "不得照抄、续写或复述，更不要把旧的长篇瓜或新闻接着讲下去。"
                            "本轮必须回答最后的【当前评论】。如果当前评论正在纠正"
                            "历史里的称呼、身份、事实或误解，应接受本轮纠正并据此回答，不能固守旧说法。"
                        )
                        context_sections.append(f"【历史记忆，仅供理解】\n{memory}")
                    if active_topic:
                        instructions += (
                            "用户可能在延续刚才的播报；依据附带话题回答，涉及新进展仍须查询。"
                        )
                        context_sections.append(active_topic)
                    if prefetched:
                        context_sections.append(f"【已查到的资料】\n{prefetched}")
                    if request.avoid_reply:
                        instructions += (
                            "上一次生成因重复旧答案已被系统拦截。重新理解当前评论，换一个直接、相关的回答；"
                            "禁止复用所附旧答案的开头、句式和结论。"
                        )
                        context_sections.append(
                            f"【禁止复用的旧答案】\n{request.avoid_reply}"
                        )
                    context_sections.append(
                        f"【当前评论，这是唯一需要回答的问题】\n"
                        f"直播间观众“{request.speaker}”说：{request.prompt}"
                    )
                    # Put the current request last. Small local models strongly
                    # weight the tail of a prompt; appending retrieved history
                    # after it previously caused the old answer to be replayed.
                    user_text = "\n\n".join(context_sections)
                pending_calls: list[dict[str, str]] = []
                leftover_discovery_calls: list[dict[str, str]] = []
                tool_rounds = 0
                discovery_rounds = 0
                answer_retries = 0
                requested_capabilities: set[str] = set()
                progress_spoken = False
                usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                session: dict[str, Any] = {
                    "type": "realtime",
                    "instructions": instructions,
                    "audio": {"output": {"voice": "active_profile"}},
                }
                if public_tools:
                    session["instructions"] += (
                        "普通闲聊、情绪表达、吐槽、承接上下文和角色互动直接回答，不调用工具。"
                        "只有用户明确要求查询，或问题确实依赖最新外部事实时，才调用"
                        f" {web_tool_name}。不要申请其他能力，也不要调用其他工具。"
                        "外部查询取得证据后必须在本轮给出结论。"
                    )
                    session.update(tools=public_tools, tool_choice="auto")
                await ws.send(
                    json.dumps({"type": "session.update", "session": session}, ensure_ascii=False)
                )
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
                    if event_type in {
                        "response.audio_transcript.delta",
                        "response.output_audio_transcript.delta",
                    }:
                        segment_delta += str(event.get("delta") or "")
                        transcript = join_spoken(completed_transcript, segment_delta)
                        round_finalized = False
                        now = time.monotonic()
                        if now - last_partial_at >= 0.12 or re.search(r"[。！？!?]\s*$", transcript):
                            await publish_spoken(partial=True)
                    elif event_type in {
                        "response.audio_transcript.done",
                        "response.output_audio_transcript.done",
                    }:
                        segment = str(event.get("transcript") or segment_delta).strip()
                        # Some realtime backends emit the full accumulated
                        # transcript, while this pipeline emits one done event
                        # per TTS sentence. Support both without duplicating.
                        if completed_transcript and segment.startswith(completed_transcript):
                            completed_transcript = segment
                        else:
                            completed_transcript = join_spoken(completed_transcript, segment)
                        segment_delta = ""
                        transcript = completed_transcript
                        await publish_spoken(partial=False)
                    elif event_type == "response.function_call_arguments.done":
                        call_name = str(event.get("name") or "")
                        call = {
                            "name": call_name,
                            "arguments": str(event.get("arguments") or "{}"),
                            "call_id": str(event.get("call_id") or ""),
                        }
                        if web_tool and call_name == web_tool_name:
                            active_external_tools[call_name] = web_tool
                            pending_calls.append(call)
                        elif call_name == self.mcp_gateway.DISCOVERY_TOOL_NAME:
                            leftover_discovery_calls.append(call)
                            logger.info(
                                "ignored leftover discovery tool call request=%s",
                                request.message_id,
                            )
                        else:
                            logger.warning(
                                "ignored undeclared model tool call name=%r request=%s",
                                call_name,
                                request.message_id,
                            )
                    elif event_type == "response.done":
                        response = event.get("response") if isinstance(event.get("response"), dict) else {}
                        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
                        for key in usage_totals:
                            try:
                                usage_totals[key] += max(0, int(usage.get(key) or 0))
                            except (TypeError, ValueError):
                                pass
                        cancelled = response.get("status") in {"cancelled", "failed", "incomplete"}
                        if transcript.strip() and (not round_finalized or cancelled):
                            await publish_spoken(partial=False, interrupted=cancelled)
                        if leftover_discovery_calls:
                            discovery_rounds += 1
                            ignored, leftover_discovery_calls = leftover_discovery_calls, []
                            output = json.dumps({
                                "enabled": [web_tool_name] if web_tool_name else [],
                                "instruction": (
                                    f"普通闲聊直接回答。只有要查网时才调用 {web_tool_name}。"
                                    if web_tool_name else
                                    "直接自然回答，不要再申请其他能力。"
                                ),
                            }, ensure_ascii=False)
                            session.update(tools=public_tools, tool_choice="auto" if public_tools else "none")
                            await ws.send(json.dumps(
                                {"type": "session.update", "session": session}, ensure_ascii=False
                            ))
                            for call in ignored:
                                await ws.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": call["call_id"],
                                        "output": output,
                                    },
                                }, ensure_ascii=False))
                            if not pending_calls:
                                await ws.send(json.dumps({"type": "response.create", "response": {}}))
                                speech_round += 1
                                transcript = ""
                                completed_transcript = ""
                                segment_delta = ""
                                round_finalized = False
                                last_partial_at = 0.0
                                continue
                        if pending_calls and tool_rounds < 3:
                            tool_rounds += 1
                            calls, pending_calls = pending_calls, []
                            candidates = [tool_progress.get(call["name"], "") for call in calls]
                            progress_text = next((text for text in candidates if text), "正在查询并核对资料")
                            if len(calls) > 1:
                                progress_text = "我正在同时查几个来源，核对清楚就告诉你呀。"

                            async def play_progress() -> None:
                                if job is not None:
                                    job.update(
                                        phase="researching", status_text=progress_text,
                                        feedback_count=int(job.get("feedback_count") or 0) + 1,
                                        terminal=False,
                                    )
                                    await self.room.publish_agent_job(job, reply_to=reply_quote)

                            async def execute_call(call: dict[str, str]) -> tuple[dict[str, str], str]:
                                try:
                                    if call["name"] not in active_external_tools:
                                        raise PermissionError("工具未在本轮服务端许可清单中")
                                    arguments = json.loads(call["arguments"])
                                    output = await self.mcp_gateway.call(call["name"], arguments)
                                except Exception as exc:  # noqa: BLE001
                                    output = f"工具调用失败：{exc}"
                                return call, output

                            execution = asyncio.gather(*(execute_call(call) for call in calls))
                            if progress_spoken:
                                pass
                            elif not transcript.strip():
                                # Start the tools first, then speak while their
                                # network calls are already in flight.
                                await play_progress()
                            else:
                                # The model already gave an immediate update.
                                # Add one concrete capability update only when
                                # the actual work is taking noticeably longer.
                                done, _ = await asyncio.wait({execution}, timeout=2.5)
                                if not done:
                                    await play_progress()
                            executed = await execution
                            for call, output in executed:
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
                            # Tool evidence is now present.  Release the tool-call
                            # constraint so the following turn can give the actual
                            # answer (or choose one bounded follow-up lookup).
                            session["tool_choice"] = "auto"
                            session["instructions"] += (
                                "资料已返回。用两三句中文口语讲最有意思的一点就够。"
                                f"{SPOKEN_CHINESE_POLICY}"
                            )
                            await ws.send(json.dumps(
                                {"type": "session.update", "session": session},
                                ensure_ascii=False,
                            ))
                            await ws.send(json.dumps({"type": "response.create", "response": {}}))
                            speech_round += 1
                            transcript = ""
                            completed_transcript = ""
                            segment_delta = ""
                            round_finalized = False
                            last_partial_at = 0.0
                            continue
                        if (
                            public_tools
                            and tool_rounds == 0
                            and looks_like_deferred_answer(transcript)
                            and answer_retries < 2
                        ):
                            answer_retries += 1
                            speech_round += 1
                            transcript = ""
                            completed_transcript = ""
                            segment_delta = ""
                            round_finalized = False
                            last_partial_at = 0.0
                            session["tool_choice"] = "required"
                            session["instructions"] += (
                                f"不要再说稍等。现在立刻调用 {web_tool_name}；"
                                "不要回复文字，也不要凭记忆回答。"
                            )
                            await ws.send(json.dumps(
                                {"type": "session.update", "session": session},
                                ensure_ascii=False,
                            ))
                            await ws.send(json.dumps({"type": "response.create", "response": {}}))
                            continue
                        if (
                            tool_rounds > 0
                            and looks_like_deferred_answer(transcript)
                            and answer_retries < 2
                        ):
                            answer_retries += 1
                            speech_round += 1
                            transcript = ""
                            completed_transcript = ""
                            segment_delta = ""
                            round_finalized = False
                            last_partial_at = 0.0
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "message",
                                            "role": "user",
                                            "content": [{
                                                "type": "input_text",
                                                "text": (
                                                    "查询已经完成。不要说你要去查或让观众等待；"
                                                    "请立刻根据上面的资料给出具体结论和依据。"
                                                ),
                                            }],
                                        },
                                    },
                                    ensure_ascii=False,
                                )
                            )
                            await ws.send(json.dumps({"type": "response.create", "response": {}}))
                            continue
                        if tool_rounds > 0 and looks_like_deferred_answer(transcript):
                            raise RuntimeError("模型连续生成了查询承诺，而不是最终答案")
                        if (
                            prefetched
                            and looks_like_deferred_answer(transcript)
                            and answer_retries < 2
                        ):
                            answer_retries += 1
                            speech_round += 1
                            transcript = ""
                            completed_transcript = ""
                            segment_delta = ""
                            round_finalized = False
                            last_partial_at = 0.0
                            session["instructions"] += (
                                "资料已经在上面。不要说正在查、正在联网或稍等。"
                                "立刻根据【已查到的资料】用中文口语说一两件具体的事。"
                            )
                            await ws.send(json.dumps(
                                {"type": "session.update", "session": session},
                                ensure_ascii=False,
                            ))
                            await ws.send(json.dumps({"type": "response.create", "response": {}}))
                            continue
                        if prefetched and looks_like_deferred_answer(transcript):
                            raise RuntimeError("模型连续生成了查询承诺，而不是最终答案")
                        if looks_like_english_answer(transcript) and answer_retries < 2:
                            answer_retries += 1
                            speech_round += 1
                            transcript = ""
                            completed_transcript = ""
                            segment_delta = ""
                            round_finalized = False
                            last_partial_at = 0.0
                            session["instructions"] += (
                                "上一句说成英文了。立刻用中文口语重说，不要英文，不要Markdown。"
                            )
                            await ws.send(json.dumps(
                                {"type": "session.update", "session": session},
                                ensure_ascii=False,
                            ))
                            await ws.send(json.dumps({"type": "response.create", "response": {}}))
                            continue
                        if looks_like_english_answer(transcript):
                            raise RuntimeError("模型用英文回答了中文观众")
                        if not transcript.strip():
                            if tool_rounds > 0:
                                transcript = (
                                    "网这会儿连得慢，我没把这条查完整。"
                                    "你等一会儿再叫我查一次就好。"
                                )
                                await publish_spoken(partial=False)
                            else:
                                raise RuntimeError("模型没有生成可播报的最终答案")
                        if request.proactive:
                            store = getattr(self.room, "store", None)
                            if store is not None:
                                await store.finalize_active_news_broadcast(
                                    transcript, f"{speech_base_id}:{speech_round}"
                                )
                        elif job is not None:
                            metrics = dict(job.get("metrics") or {})
                            metrics.update({
                                "usage": usage_totals,
                                "tool_rounds": tool_rounds,
                                "discovery_rounds": discovery_rounds,
                                "exposed_tool_count": len(public_tools),
                                "requested_capabilities": sorted(requested_capabilities),
                            })
                            job.update(
                                phase="completed", status_text=transcript, final_text=transcript,
                                terminal=True, error="", tool_rounds=tool_rounds, metrics=metrics,
                            )
                            await self.room.publish_agent_job(job, reply_to=reply_quote)
                        return
        finally:
            if prefetch_task is not None and not prefetch_task.done():
                prefetch_task.cancel()
            if transcript.strip() and not round_finalized:
                with contextlib.suppress(Exception):
                    await publish_spoken(partial=False, interrupted=True)

    async def _speak_exact(self, ws, phrase: str) -> None:
        """Speak already-approved text without exposing a fresh generative draft."""
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": (
                    "只逐字朗读用户提供的文字，不要回答、改写、解释、调用工具或增加任何内容。"
                ),
                "audio": {"output": {"voice": "active_profile"}},
            },
        }, ensure_ascii=False))
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{
                "type": "input_text", "text": f"逐字朗读：{phrase}",
            }]},
        }, ensure_ascii=False))
        await ws.send(json.dumps({"type": "response.create", "response": {}}, ensure_ascii=False))
        while True:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
            if event.get("type") == "response.done":
                response = event.get("response") if isinstance(event.get("response"), dict) else {}
                if response.get("status") in {"failed", "cancelled", "incomplete"}:
                    raise RuntimeError("过程语音没有完整播放")
                return

    @staticmethod
    def _reply_quote(request: MentionRequest) -> dict[str, str] | None:
        if request.proactive or request.welcome:
            return None
        return {
            "id": request.message_id,
            "participant_id": request.participant_id,
            "speaker": request.speaker,
            "text": request.text,
        }
