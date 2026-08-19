"""Single-room audience presence and fair call-in queue.

The avatar FLV stream is shared by every viewer, while exactly one participant
may own the speech-to-speech WebSocket. State is intentionally in memory: this
deployment runs one Uvicorn worker and a restart should clear stale callers.
"""

from __future__ import annotations

import asyncio
import random
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any


CHINESE_SURNAMES = (
    "林", "苏", "沈", "顾", "陆", "江", "叶", "夏", "白", "许", "周", "程",
    "唐", "宋", "温", "秦", "楚", "宁", "乔", "简", "洛", "黎", "季", "云",
)
CHINESE_GIVEN = (
    "星河", "清欢", "知夏", "听澜", "若安", "予晴", "景行", "晚舟", "念初", "时雨",
    "安歌", "南乔", "月白", "青禾", "书昀", "映雪", "云舒", "朝颜", "凌霄", "亦辰",
)
ENGLISH_NAMES = (
    "Avery", "Blake", "Casey", "Dylan", "Emery", "Finley", "Harper", "Jamie",
    "Jordan", "Kai", "Logan", "Morgan", "Nova", "Parker", "Quinn", "Reese",
    "Riley", "Robin", "Rowan", "Sage", "Skyler", "Taylor", "Winter", "Zion",
)
_SPACE_RE = re.compile(r"\s+")
_REASONING_BLOCK_RE = re.compile(
    r"<(think|analysis|reasoning)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_REASONING_OPEN_RE = re.compile(r"<(?:think|analysis|reasoning)\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_REASONING_CLOSE_RE = re.compile(r"</(?:think|analysis|reasoning)\s*>", re.IGNORECASE)
MESSAGE_LIMIT = 80
CHAT_WINDOW_SECONDS = 10.0
CHAT_WINDOW_MESSAGES = 5


def _clean_public_text(value: str) -> str:
    """Remove provider reasoning markup before it reaches the public room."""
    text = str(value or "")
    text = _REASONING_BLOCK_RE.sub("", text)
    text = _REASONING_OPEN_RE.sub("", text)
    text = _REASONING_CLOSE_RE.sub("", text)
    return _SPACE_RE.sub(" ", text).strip()


class RoomError(RuntimeError):
    def __init__(self, message: str, *, status: int = 400, code: str = "room_error"):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(slots=True)
class Participant:
    id: str
    token: str
    name_zh: str
    name_en: str
    display_name: str
    created_at: float
    last_seen: float
    connections: int = 0


@dataclass(slots=True)
class QueueEntry:
    id: str
    participant_token: str
    created_at: float
    last_poll: float


@dataclass(slots=True)
class CallSession:
    id: str
    participant_token: str
    websocket_ticket: str
    granted_at: float
    pending_expires_at: float
    connected_at: float | None = None


class LiveRoom:
    def __init__(
        self,
        *,
        queue_limit: int = 100,
        pending_timeout_s: int = 60,
        queue_ttl_s: int = 30 * 60,
        disconnect_grace_s: int = 12,
        max_call_s: int = 10 * 60,
    ) -> None:
        self.queue_limit = max(1, queue_limit)
        self.pending_timeout_s = max(10, pending_timeout_s)
        self.queue_ttl_s = max(60, queue_ttl_s)
        self.disconnect_grace_s = max(3, disconnect_grace_s)
        self.max_call_s = max(60, max_call_s)
        self._participants: dict[str, Participant] = {}
        self._queue: list[QueueEntry] = []
        self._active: CallSession | None = None
        self._messages: list[dict[str, Any]] = []
        self._chat_times: dict[str, list[float]] = {}
        self._subscribers: dict[asyncio.Queue, str] = {}
        self._disconnect_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._sweeper_task: asyncio.Task | None = None

    def start(self) -> None:
        if self._sweeper_task is None or self._sweeper_task.done():
            self._sweeper_task = asyncio.create_task(self._sweeper())

    async def stop(self) -> None:
        if self._sweeper_task:
            self._sweeper_task.cancel()
            try:
                await self._sweeper_task
            except asyncio.CancelledError:
                pass
            self._sweeper_task = None

    async def identify(self, token: str | None, *, create: bool = True) -> tuple[Participant, bool]:
        async with self._lock:
            participant = self._participants.get(token or "")
            if participant is not None:
                participant.last_seen = time.monotonic()
                return participant, False
            if not create:
                raise RoomError("请先进入直播间", status=401, code="not_joined")
            participant = self._new_participant()
            self._participants[participant.token] = participant
            self._broadcast_locked()
            return participant, True

    def _new_participant(self) -> Participant:
        now = time.monotonic()
        for _ in range(30):
            if random.choice((True, False)):
                zh = f"{random.choice(CHINESE_SURNAMES)}{random.choice(CHINESE_GIVEN)}"
                en = ""
                display = zh
            else:
                zh = ""
                first = random.choice(ENGLISH_NAMES)
                second = random.choice(tuple(name for name in ENGLISH_NAMES if name != first))
                en = f"{first} {second}"
                display = en
            if all(p.display_name.casefold() != display.casefold() for p in self._participants.values()):
                break
        else:
            if zh:
                zh = f"{zh}{random.choice(CHINESE_GIVEN)}"
                display = zh
            else:
                en = f"{en} {secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
                display = en
        return Participant(
            id=secrets.token_hex(6),
            token=secrets.token_urlsafe(32),
            name_zh=zh,
            name_en=en,
            display_name=display,
            created_at=now,
            last_seen=now,
        )

    async def rename(self, token: str, raw_name: str) -> dict[str, Any]:
        name = _SPACE_RE.sub(" ", str(raw_name or "")).strip()
        if not 1 <= len(name) <= 24:
            raise RoomError("名字长度需要在 1 到 24 个字符之间")
        if any(ord(ch) < 32 for ch in name):
            raise RoomError("名字包含无效字符")
        async with self._lock:
            participant = self._require_locked(token)
            if any(
                p.token != token and p.display_name.casefold() == name.casefold()
                for p in self._participants.values()
            ):
                raise RoomError("这个名字已经有人使用", status=409, code="name_taken")
            participant.display_name = name
            participant.last_seen = time.monotonic()
            self._broadcast_locked()
            return self._participant_public(participant)

    async def subscribe(self, token: str) -> asyncio.Queue:
        async with self._lock:
            participant = self._require_locked(token)
            participant.connections += 1
            participant.last_seen = time.monotonic()
            task = self._disconnect_tasks.pop(token, None)
            if task:
                task.cancel()
            channel: asyncio.Queue = asyncio.Queue(maxsize=4)
            self._subscribers[channel] = token
            channel.put_nowait(self._snapshot_locked(token))
            self._broadcast_locked(skip=channel)
            return channel

    async def unsubscribe(self, channel: asyncio.Queue) -> None:
        async with self._lock:
            token = self._subscribers.pop(channel, None)
            if not token:
                return
            participant = self._participants.get(token)
            if participant:
                participant.connections = max(0, participant.connections - 1)
                participant.last_seen = time.monotonic()
                if participant.connections == 0:
                    old = self._disconnect_tasks.pop(token, None)
                    if old:
                        old.cancel()
                    self._disconnect_tasks[token] = asyncio.create_task(self._disconnect_after_grace(token))
            self._broadcast_locked()

    async def _disconnect_after_grace(self, token: str) -> None:
        this_task = asyncio.current_task()
        try:
            await asyncio.sleep(self.disconnect_grace_s)
            async with self._lock:
                participant = self._participants.get(token)
                if not participant or participant.connections:
                    return
                changed = self._remove_queue_token_locked(token)
                if self._active and self._active.participant_token == token and self._active.connected_at is None:
                    self._active = None
                    changed = True
                if changed:
                    self._broadcast_locked()
        except asyncio.CancelledError:
            pass
        finally:
            # A rapid disconnect -> reconnect -> disconnect can already have
            # installed a newer grace timer for this token. The cancelled old
            # timer must not remove that newer task from the registry.
            async with self._lock:
                if self._disconnect_tasks.get(token) is this_task:
                    self._disconnect_tasks.pop(token, None)

    async def request_session(self, token: str) -> dict[str, Any]:
        async with self._lock:
            participant = self._require_locked(token)
            participant.last_seen = time.monotonic()
            self._expire_active_locked()

            if self._active and self._active.participant_token == token:
                if self._active.connected_at is not None or not self._active.websocket_ticket:
                    raise RoomError("你已经在连线中", status=409, code="already_active")
                return {"state": "granted", **self._grant_public_locked(self._active)}

            existing = self._queue_for_token_locked(token)
            if existing:
                return self._queued_public_locked(existing)

            # Fairness: a newly arriving participant never jumps a queue whose
            # first member is waiting to claim a newly freed slot.
            if self._active is None and not self._queue:
                session = self._grant_locked(token)
                self._broadcast_locked()
                return {"state": "granted", **self._grant_public_locked(session)}

            if len(self._queue) >= self.queue_limit:
                raise RoomError("连线队列已满，请稍后再试", status=503, code="at_capacity")
            now = time.monotonic()
            entry = QueueEntry(
                id=secrets.token_urlsafe(18),
                participant_token=token,
                created_at=now,
                last_poll=now,
            )
            self._queue.append(entry)
            self._broadcast_locked()
            return self._queued_public_locked(entry)

    async def poll_queue(self, token: str, queue_id: str) -> dict[str, Any]:
        async with self._lock:
            self._require_locked(token).last_seen = time.monotonic()
            self._expire_active_locked()
            entry = next((item for item in self._queue if item.id == queue_id), None)
            if not entry or entry.participant_token != token:
                raise RoomError("排队凭证已失效", status=404, code="queue_expired")
            entry.last_poll = time.monotonic()
            if self._active is None and self._queue and self._queue[0] is entry:
                self._queue.pop(0)
                session = self._grant_locked(token)
                self._broadcast_locked()
                return {"state": "granted", **self._grant_public_locked(session)}
            return self._queued_public_locked(entry)

    async def leave_queue(self, token: str, queue_id: str | None = None) -> bool:
        async with self._lock:
            self._require_locked(token)
            before = len(self._queue)
            self._queue = [
                item for item in self._queue
                if not (item.participant_token == token and (not queue_id or item.id == queue_id))
            ]
            changed = len(self._queue) != before
            if changed:
                self._broadcast_locked()
            return changed

    async def claim_websocket(self, token: str, ticket: str) -> tuple[str, str]:
        async with self._lock:
            participant = self._require_locked(token)
            self._expire_active_locked()
            session = self._active
            if (
                not session
                or session.participant_token != token
                or not ticket
                or not secrets.compare_digest(session.websocket_ticket, ticket)
            ):
                raise RoomError("连线票据无效或已过期", status=403, code="ticket_invalid")
            session.websocket_ticket = ""
            session.connected_at = time.monotonic()
            participant.last_seen = time.monotonic()
            self._broadcast_locked()
            return session.id, participant.display_name

    async def end_session(self, token: str, session_id: str | None = None) -> bool:
        async with self._lock:
            session = self._active
            if not session or session.participant_token != token:
                return False
            if session_id and session.id != session_id:
                return False
            self._active = None
            self._broadcast_locked()
            return True

    async def snapshot(self, token: str) -> dict[str, Any]:
        async with self._lock:
            self._require_locked(token).last_seen = time.monotonic()
            if self._expire_active_locked():
                self._broadcast_locked()
            return self._snapshot_locked(token)

    async def publish_transcript(
        self,
        *,
        session_id: str,
        event_id: str,
        role: str,
        speaker: str,
        text: str,
        partial: bool = False,
        interrupted: bool = False,
    ) -> None:
        """Upsert a recent public-room transcript line and notify all viewers."""
        clean = _clean_public_text(text)[:500]
        if not clean or role not in {"user", "assistant"}:
            return
        message_id = f"{session_id}:{role}:{event_id or 'unknown'}"
        async with self._lock:
            message = next((item for item in self._messages if item["id"] == message_id), None)
            if message is None:
                message = {
                    "id": message_id,
                    "kind": "voice",
                    "role": role,
                    "speaker": speaker,
                    "text": clean,
                    "partial": bool(partial),
                    "interrupted": bool(interrupted),
                    "created_at": time.time(),
                }
                self._messages.append(message)
                self._messages = self._messages[-MESSAGE_LIMIT:]
            else:
                message.update(
                    text=clean,
                    speaker=speaker,
                    partial=bool(partial),
                    interrupted=bool(interrupted),
                )
            self._broadcast_locked()

    async def publish_chat(self, token: str, raw_text: str) -> dict[str, Any]:
        """Publish a rate-limited viewer text message to the whole room."""
        clean = _SPACE_RE.sub(" ", str(raw_text or "")).strip()
        if not clean:
            raise RoomError("消息不能为空", code="empty_message")
        if len(clean) > 200:
            raise RoomError("消息最多 200 个字符", code="message_too_long")
        if any(ord(ch) < 32 for ch in clean):
            raise RoomError("消息包含无效字符", code="invalid_message")
        now = time.monotonic()
        async with self._lock:
            participant = self._require_locked(token)
            participant.last_seen = now
            recent = [
                sent_at for sent_at in self._chat_times.get(token, [])
                if now - sent_at < CHAT_WINDOW_SECONDS
            ]
            if len(recent) >= CHAT_WINDOW_MESSAGES:
                raise RoomError("发送太快了，请稍等一下", status=429, code="rate_limited")
            recent.append(now)
            self._chat_times[token] = recent
            message = {
                "id": f"chat:{secrets.token_urlsafe(12)}",
                "kind": "chat",
                "role": "viewer",
                "participant_id": participant.id,
                "speaker": participant.display_name,
                "text": clean,
                "partial": False,
                "interrupted": False,
                "created_at": time.time(),
            }
            self._messages.append(message)
            self._messages = self._messages[-MESSAGE_LIMIT:]
            self._broadcast_locked()
            return dict(message)

    async def can_bot_reply(self) -> bool:
        """The room bot may speak only when no call grant/queue owns priority."""
        async with self._lock:
            self._expire_active_locked()
            return self._active is None and not self._queue

    async def publish_bot_reply(
        self,
        *,
        message_id: str,
        text: str,
        reply_to: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Publish one final assistant reply linked to the triggering comment."""
        clean = _clean_public_text(text)[:500]
        if not clean:
            return None
        quote = {
            "id": str(reply_to.get("id") or ""),
            "speaker": str(reply_to.get("speaker") or "观众")[:24],
            "text": _clean_public_text(str(reply_to.get("text") or ""))[:120],
        }
        async with self._lock:
            item = {
                "id": f"bot:{message_id}",
                "kind": "mention_reply",
                "role": "assistant",
                "speaker": "小麻",
                "text": clean,
                "reply_to": quote,
                "partial": False,
                "interrupted": False,
                "created_at": time.time(),
            }
            existing = next((index for index, old in enumerate(self._messages) if old["id"] == item["id"]), None)
            if existing is None:
                self._messages.append(item)
            else:
                self._messages[existing] = item
            self._messages = self._messages[-MESSAGE_LIMIT:]
            self._broadcast_locked()
            return dict(item)

    def _grant_locked(self, token: str) -> CallSession:
        now = time.monotonic()
        self._active = CallSession(
            id=secrets.token_urlsafe(18),
            participant_token=token,
            websocket_ticket=secrets.token_urlsafe(32),
            granted_at=now,
            pending_expires_at=now + self.pending_timeout_s,
        )
        return self._active

    def _grant_public_locked(self, session: CallSession) -> dict[str, Any]:
        return {
            "session_id": session.id,
            "session_token": session.websocket_ticket,
            "pending_timeout_s": self.pending_timeout_s,
            "tier": "room",
            "limited": False,
        }

    def _queued_public_locked(self, entry: QueueEntry) -> dict[str, Any]:
        try:
            position = self._queue.index(entry) + 1
        except ValueError:
            position = 0
        return {
            "state": "queued",
            "queue_id": entry.id,
            "position": position,
            "poll_interval_s": 2,
            "tier": "room",
        }

    def _snapshot_locked(self, token: str) -> dict[str, Any]:
        participant = self._participants.get(token)
        active_participant = (
            self._participants.get(self._active.participant_token) if self._active else None
        )
        queue_public = []
        my_position = 0
        for index, entry in enumerate(self._queue, 1):
            queued = self._participants.get(entry.participant_token)
            if not queued:
                continue
            item = {**self._participant_public(queued), "position": index}
            queue_public.append(item)
            if queued.token == token:
                my_position = index
        is_active = bool(self._active and self._active.participant_token == token)
        status = "calling" if is_active and self._active and self._active.connected_at else (
            "ready" if is_active else ("queued" if my_position else "watching")
        )
        viewer_tokens = {
            p.token for p in self._participants.values() if p.connections > 0
        }
        if self._active:
            viewer_tokens.add(self._active.participant_token)
        queued_tokens = {entry.participant_token for entry in self._queue}
        viewers = []
        for viewer in sorted(
            (p for p in self._participants.values() if p.token in viewer_tokens),
            key=lambda p: (
                0 if self._active and p.token == self._active.participant_token else 1,
                p.created_at,
            ),
        ):
            item = self._participant_public(viewer)
            item["status"] = (
                ("calling" if self._active.connected_at else "ready")
                if self._active and viewer.token == self._active.participant_token
                else ("queued" if viewer.token in queued_tokens else "watching")
            )
            viewers.append(item)
        return {
            "room": "default",
            "viewer_count": len(viewer_tokens),
            "viewers": viewers,
            "queue_limit": self.queue_limit,
            "queue": queue_public,
            "active": self._participant_public(active_participant) if active_participant else None,
            "messages": [dict(item) for item in self._messages],
            "me": {
                **(self._participant_public(participant) if participant else {}),
                "status": status,
                "queue_position": my_position,
            },
        }

    @staticmethod
    def _participant_public(participant: Participant | None) -> dict[str, Any]:
        if participant is None:
            return {}
        return {
            "id": participant.id,
            "name": participant.display_name,
            "name_zh": participant.name_zh,
            "name_en": participant.name_en,
        }

    def _require_locked(self, token: str) -> Participant:
        participant = self._participants.get(token)
        if not participant:
            raise RoomError("直播间身份已失效，请刷新页面", status=401, code="not_joined")
        return participant

    def _queue_for_token_locked(self, token: str) -> QueueEntry | None:
        return next((item for item in self._queue if item.participant_token == token), None)

    def _remove_queue_token_locked(self, token: str) -> bool:
        before = len(self._queue)
        self._queue = [item for item in self._queue if item.participant_token != token]
        return before != len(self._queue)

    def _expire_active_locked(self) -> bool:
        if self._active and self._active.connected_at is None and time.monotonic() >= self._active.pending_expires_at:
            self._active = None
            return True
        return False

    def _broadcast_locked(self, *, skip: asyncio.Queue | None = None) -> None:
        stale: list[asyncio.Queue] = []
        for channel, token in tuple(self._subscribers.items()):
            if channel is skip:
                continue
            payload = self._snapshot_locked(token)
            if channel.full():
                try:
                    channel.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                channel.put_nowait(payload)
            except asyncio.QueueFull:
                stale.append(channel)
        for channel in stale:
            self._subscribers.pop(channel, None)

    async def _sweeper(self) -> None:
        while True:
            await asyncio.sleep(5)
            async with self._lock:
                now = time.monotonic()
                changed = self._expire_active_locked()
                before = len(self._queue)
                self._queue = [item for item in self._queue if now - item.last_poll < self.queue_ttl_s]
                changed = changed or before != len(self._queue)
                protected = {item.participant_token for item in self._queue}
                if self._active:
                    protected.add(self._active.participant_token)
                for token, participant in tuple(self._participants.items()):
                    if participant.connections == 0 and token not in protected and now - participant.last_seen > 3600:
                        self._participants.pop(token, None)
                        self._chat_times.pop(token, None)
                if changed:
                    self._broadcast_locked()
