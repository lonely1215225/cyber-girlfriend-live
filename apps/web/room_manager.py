"""Single-room audience presence and fair call-in queue.

The avatar FLV stream is shared by every viewer, while exactly one participant
may own the speech-to-speech WebSocket. State is intentionally in memory: this
deployment runs one Uvicorn worker and a restart should clear stale callers.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any

from room_store import RoomStore

try:
    from fish_s2_tags import clean_public_fish_text
except ImportError:
    _FISH_TAG_RE = re.compile(r"\[[A-Za-z][^\[\]]{0,80}\]")

    def clean_public_fish_text(text: str) -> str:
        value = _FISH_TAG_RE.sub("", str(text or ""))
        opening = value.rfind("[")
        if opening >= 0:
            tail = value[opening:]
            if "]" not in tail and re.match(r"\[[A-Za-z]", tail):
                value = value[:opening]
        return value


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
_PRIVATE_PROTOCOL_BLOCK_RE = re.compile(
    r"<(tool_call|toolcall|function_call|functioncall)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_PRIVATE_PROTOCOL_OPEN_RE = re.compile(
    r"<(?:tool_call|toolcall|function_call|functioncall)\b[^>]*>.*$",
    re.IGNORECASE | re.DOTALL,
)
_PRIVATE_PROTOCOL_CLOSE_RE = re.compile(
    r"</(?:tool_call|toolcall|function_call|functioncall)\s*>", re.IGNORECASE
)
# happy/serious remain here only so leftover model tags stay hidden from chat.
_DELIVERY_PROFILES = (
    "neutral", "happy", "surprised", "serious", "pout", "one_brow",
    "smirk", "wink", "cheek_puff", "cute_annoyed", "shy", "laugh",
)
_DELIVERY_STYLES = ("neutral", "gentle", "calm", "cheerful", "serious")
_VOCAL_EMOTIONS = (
    "neutral", "happy", "playful", "warm", "tender", "shy", "serious",
    "sad", "angry", "surprised",
)
_NONVERBAL_EVENTS = ("none", "soft_laugh", "laugh", "sigh", "breath", "hum")
_DELIVERY_TAG_RE = re.compile(r"</?e(?:\s+[^>]*)?>", re.IGNORECASE)
_DELIVERY_COMPACT_BARE_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:{'|'.join(sorted(_DELIVERY_PROFILES, key=len, reverse=True))})"
    rf"\s+(?:0(?:\.\d+)?|1(?:\.0+)?)"
    rf"\s+(?:{'|'.join(sorted(_VOCAL_EMOTIONS, key=len, reverse=True))})"
    rf"\s+(?:0(?:\.\d+)?|1(?:\.0+)?)"
    rf"\s+(?:{'|'.join(sorted(_NONVERBAL_EVENTS, key=len, reverse=True))})"
    rf"\s+(?:0(?:\.\d+)?|1(?:\.\d+)?)"
    r"(?=$|[\s，。！？、；：,.!?;:])",
    re.IGNORECASE,
)
_DELIVERY_BARE_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:{'|'.join(sorted(_DELIVERY_PROFILES, key=len, reverse=True))})"
    rf"\s+(?:0(?:\.\d+)?|1(?:\.0+)?)"
    rf"\s+(?:{'|'.join(sorted(_DELIVERY_STYLES, key=len, reverse=True))})"
    rf"(?:\s+(?:0(?:\.\d+)?|1(?:\.0+)?))?"
    r"(?=$|[\s，。！？、；：,.!?;:])",
    re.IGNORECASE,
)
_DELIVERY_PROFILE_SET = frozenset(_DELIVERY_PROFILES)
_DELIVERY_STYLE_SET = frozenset(_DELIVERY_STYLES)
_VOCAL_SET = frozenset(_VOCAL_EMOTIONS)
_NONVERBAL_SET = frozenset(_NONVERBAL_EVENTS)
MESSAGE_LIMIT = 80
CHAT_WINDOW_SECONDS = 10.0
CHAT_WINDOW_MESSAGES = 5


def _clean_public_text(value: str, *, assistant: bool = False) -> str:
    """Remove provider reasoning markup before it reaches the public room."""
    text = str(value or "")
    if assistant:
        text = _PRIVATE_PROTOCOL_BLOCK_RE.sub("", text)
        text = _PRIVATE_PROTOCOL_OPEN_RE.sub("", text)
        text = _PRIVATE_PROTOCOL_CLOSE_RE.sub("", text)
    text = _REASONING_BLOCK_RE.sub("", text)
    text = _REASONING_OPEN_RE.sub("", text)
    text = _REASONING_CLOSE_RE.sub("", text)
    if assistant:
        text = _DELIVERY_TAG_RE.sub("", text)
        opening = text.rfind("<")
        if opening >= 0:
            tail = text[opening:]
            if ">" not in tail and re.match(r"</?e\b|^<$", tail, re.IGNORECASE):
                text = text[:opening]
        text = _DELIVERY_COMPACT_BARE_RE.sub("", text)
        text = _DELIVERY_BARE_RE.sub("", text)
        # Partial transcript events can arrive token by token. Hide an exact
        # protocol prefix at the tail until it either becomes a complete
        # record (removed above) or proves to be ordinary prose.
        search_start = max(0, len(text) - 180)
        for token in re.finditer(r"(?<![A-Za-z0-9_])[A-Za-z_]", text[search_start:]):
            index = search_start + token.start()
            if _looks_like_incomplete_delivery(text[index:].split()):
                text = text[:index]
                break
        text = clean_public_fish_text(text)
    return _SPACE_RE.sub(" ", text).strip()


def _looks_like_incomplete_delivery(parts: list[str]) -> bool:
    """Whether a trailing token list can still become a hidden control record."""

    if not parts or len(parts) > 6:
        return False
    profile = parts[0].lower()
    if len(parts) == 1:
        return any(item.startswith(profile) for item in _DELIVERY_PROFILE_SET)
    if profile not in _DELIVERY_PROFILE_SET:
        return False
    if not re.fullmatch(r"[01](?:\.\d*)?", parts[1]):
        return False
    if len(parts) == 2:
        return True
    third = parts[2].lower()
    if not (
        any(item.startswith(third) for item in _DELIVERY_STYLE_SET)
        or any(item.startswith(third) for item in _VOCAL_SET)
    ):
        return False
    if len(parts) == 3:
        return True
    if not re.fullmatch(r"[01](?:\.\d*)?", parts[3]):
        return False
    if len(parts) == 4:
        return True
    nonverbal = parts[4].lower()
    if not any(item.startswith(nonverbal) for item in _NONVERBAL_SET):
        return False
    if len(parts) == 5:
        return True
    return bool(re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.\d+)?)", parts[5]))


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
        store: RoomStore | None = None,
    ) -> None:
        self.queue_limit = max(1, queue_limit)
        self.pending_timeout_s = max(10, pending_timeout_s)
        self.queue_ttl_s = max(60, queue_ttl_s)
        self.disconnect_grace_s = max(3, disconnect_grace_s)
        self.max_call_s = max(60, max_call_s)
        self.store = store
        self._participants: dict[str, Participant] = {}
        self._queue: list[QueueEntry] = []
        self._active: CallSession | None = None
        self._messages: list[dict[str, Any]] = []
        self._agent_jobs: dict[str, dict[str, Any]] = {}
        self._proactive_block_until = 0.0
        self._agent_proactive_cooldown_s = max(
            0, int(os.environ.get("AGENT_PROACTIVE_COOLDOWN_SECONDS", "60"))
        )
        self._chat_times: dict[str, list[float]] = {}
        self._subscribers: dict[asyncio.Queue, str] = {}
        self._disconnect_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._sweeper_task: asyncio.Task | None = None

    def start(self) -> None:
        if self._sweeper_task is None or self._sweeper_task.done():
            self._sweeper_task = asyncio.create_task(self._sweeper())

    async def restore(self) -> None:
        """Restore bounded public history; presence and queues stay ephemeral."""
        if not self.store:
            return
        messages = await self.store.load_recent_messages(MESSAGE_LIMIT)
        jobs = await self.store.load_agent_jobs(limit=30)
        async with self._lock:
            for message in messages:
                if message.get("role") == "assistant":
                    message["text"] = _clean_public_text(
                        str(message.get("text") or ""), assistant=True
                    )
            self._messages = messages[-MESSAGE_LIMIT:]
            self._agent_jobs = {str(item["id"]): dict(item, type="agent_job") for item in jobs}
            # Agent lifecycle data already travels in `agent_jobs`. Recreating
            # it as a second transcript message caused duplicate/stale status
            # rows after every process restart.

    async def stop(self) -> None:
        if self._sweeper_task:
            self._sweeper_task.cancel()
            try:
                await self._sweeper_task
            except asyncio.CancelledError:
                pass
            self._sweeper_task = None

    async def identify(
        self,
        token: str | None,
        *,
        create: bool = True,
        ip_address: str = "",
        user_agent: str = "",
    ) -> tuple[Participant, bool]:
        async with self._lock:
            participant = self._participants.get(token or "")
            if participant is not None:
                participant.last_seen = time.monotonic()
                return participant, False
        if token and self.store:
            saved = await self.store.resolve_user(token, ip_address=ip_address, user_agent=user_agent)
            if saved:
                now = time.monotonic()
                participant = Participant(
                    id=str(saved["id"]), token=token,
                    name_zh=str(saved.get("name_zh") or ""), name_en=str(saved.get("name_en") or ""),
                    display_name=str(saved["display_name"]), created_at=now, last_seen=now,
                )
                async with self._lock:
                    existing = self._participants.get(token)
                    if existing:
                        existing.last_seen = now
                        return existing, False
                    self._participants[token] = participant
                    self._broadcast_locked()
                return participant, False
        if not create:
            raise RoomError("请先进入直播间", status=401, code="not_joined")
        for _ in range(40):
            async with self._lock:
                participant = self._new_participant()
            if self.store and not await self.store.create_user(
                user_id=participant.id,
                display_name=participant.display_name,
                name_zh=participant.name_zh,
                name_en=participant.name_en,
                token=participant.token,
                ip_address=ip_address,
                user_agent=user_agent,
            ):
                continue
            async with self._lock:
                self._participants[participant.token] = participant
                self._broadcast_locked()
            return participant, True
        raise RoomError("暂时无法创建直播间身份", status=503, code="identity_unavailable")

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
            if self.store and not await self.store.rename_user(participant.id, name):
                raise RoomError("这个名字已经有人使用", status=409, code="name_taken")
            participant.display_name = name
            participant.last_seen = time.monotonic()
            self._broadcast_locked()
            return self._participant_public(participant)

    async def subscribe(self, token: str) -> asyncio.Queue:
        channel, _ = await self.subscribe_presence(token)
        return channel

    async def subscribe_presence(self, token: str) -> tuple[asyncio.Queue, bool]:
        """Subscribe and report a real room arrival exactly once.

        EventSource reconnects and page refreshes briefly overlap or reuse the
        disconnect grace task. Neither is a new arrival. Once the grace period
        has elapsed, the same persisted user returning is an arrival again.
        """
        async with self._lock:
            participant = self._require_locked(token)
            reconnecting = token in self._disconnect_tasks
            arrived = participant.connections == 0 and not reconnecting
            participant.connections += 1
            participant.last_seen = time.monotonic()
            task = self._disconnect_tasks.pop(token, None)
            if task:
                task.cancel()
            channel: asyncio.Queue = asyncio.Queue(maxsize=4)
            self._subscribers[channel] = token
            channel.put_nowait(self._snapshot_locked(token))
            self._broadcast_locked(skip=channel)
            return channel, arrived

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
            result = (session.id, participant.display_name)
            user_id = participant.id
        if self.store:
            await self.store.start_call(session.id, user_id, time.time(), time.time())
        return result

    async def end_session(self, token: str, session_id: str | None = None) -> bool:
        async with self._lock:
            session = self._active
            if not session or session.participant_token != token:
                return False
            if session_id and session.id != session_id:
                return False
            ended_id = session.id
            self._active = None
            self._broadcast_locked()
        if self.store:
            await self.store.end_call(ended_id)
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
        clean = _clean_public_text(text, assistant=role == "assistant")[:2000]
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
            saved = dict(message)
            saved["call_session_id"] = session_id
            user_id = self._message_user_id_locked(session_id)
        if self.store and not partial:
            await self.store.save_message(saved, user_id=user_id)

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
            saved = dict(message)
        if self.store:
            await self.store.save_message(saved, user_id=participant.id)
        return saved

    async def can_bot_reply(self) -> bool:
        """The room bot may speak only when no call grant/queue owns priority."""
        async with self._lock:
            self._expire_active_locked()
            return self._active is None and not self._queue

    def _watching_count_locked(self) -> int:
        return sum(1 for person in self._participants.values() if person.connections > 0)

    async def can_start_proactive(self) -> bool:
        """News only plays to a watched room, and never over a call or query."""
        async with self._lock:
            self._expire_active_locked()
            return (
                self._watching_count_locked() >= 1
                and self._active is None
                and not self._queue
                and time.monotonic() >= self._proactive_block_until
            )

    async def is_active_caller(self, token: str) -> bool:
        """Return whether this participant currently owns a connected call."""
        async with self._lock:
            self._require_locked(token).last_seen = time.monotonic()
            self._expire_active_locked()
            return bool(
                self._active
                and self._active.participant_token == token
                and self._active.connected_at is not None
            )

    async def publish_bot_reply(
        self,
        *,
        message_id: str,
        text: str,
        reply_to: dict[str, Any] | None,
        partial: bool = False,
        interrupted: bool = False,
        memory_user_id: str = "",
    ) -> dict[str, Any] | None:
        """Upsert a quoted comment reply or an unquoted proactive room message."""
        clean = _clean_public_text(text, assistant=True)[:2000]
        if not clean:
            return None
        quote = None
        if reply_to:
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
                "partial": partial,
                "interrupted": bool(interrupted),
                "created_at": time.time(),
            }
            if quote:
                item["reply_to"] = quote
            existing = next((index for index, old in enumerate(self._messages) if old["id"] == item["id"]), None)
            if existing is None:
                self._messages.append(item)
            else:
                self._messages[existing] = item
            self._messages = self._messages[-MESSAGE_LIMIT:]
            self._broadcast_locked()
            saved = dict(item)
            user_id = (
                str(reply_to.get("participant_id") or "") if reply_to
                else str(memory_user_id or "")
            )
        if self.store and not partial:
            await self.store.save_message(saved, user_id=user_id or None)
        return saved

    async def publish_agent_job(
        self,
        job: dict[str, Any],
        *,
        reply_to: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Persist and broadcast one public agent lifecycle update."""
        event_only = bool(job.get("event_only"))
        saved_job = await self.store.save_agent_job(job) if self.store else dict(job)
        if event_only:
            saved_job["event_only"] = True
        saved_job["type"] = "agent_job"
        job_id = str(saved_job.get("id") or job.get("id") or "")
        async with self._lock:
            self._agent_jobs[job_id] = saved_job
            # Clean up status rows produced by older server versions. Current
            # clients render lifecycle state directly from `_agent_jobs`.
            self._messages = [
                item for item in self._messages
                if not (
                    item.get("agent_job_id") == job_id
                    or item.get("id") == f"bot:agent:{job_id}"
                )
            ]
            self._proactive_block_until = max(
                self._proactive_block_until,
                time.monotonic() + self._agent_proactive_cooldown_s,
            )
            # Keep the public lifecycle list bounded even before SQLite cleanup.
            if len(self._agent_jobs) > 30:
                oldest = min(self._agent_jobs, key=lambda key: self._agent_jobs[key].get("updated_at", 0))
                self._agent_jobs.pop(oldest, None)
            if saved_job.get("event_only"):
                self._broadcast_locked()
        if saved_job.get("event_only"):
            return saved_job
        text = str(saved_job.get("final_text") or saved_job.get("status_text") or "正在处理…")
        message = await self.publish_bot_reply(
            message_id=f"agent:{job_id}", text=text, reply_to=reply_to,
            partial=not bool(saved_job.get("terminal")),
            # A failed lookup is a completed error response, not an audio
            # interruption.  Only actual cancellation should show “已打断”.
            interrupted=saved_job.get("phase") == "cancelled",
        )
        if message:
            async with self._lock:
                target = next((item for item in self._messages if item["id"] == message["id"]), None)
                if target:
                    target["kind"] = "agent_status" if not saved_job.get("terminal") else "mention_reply"
                    target["agent_job_id"] = job_id
                    target["agent_phase"] = saved_job.get("phase") or "queued"
                    self._broadcast_locked()
        return saved_job

    async def public_agent_jobs(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [dict(item) for item in self._agent_jobs.values()]

    async def memory_context(self, token: str, query: str = "") -> str:
        if not self.store:
            return ""
        async with self._lock:
            participant = self._require_locked(token)
            user_id = participant.id
        return await self.store.memory_context(user_id, query)

    async def participant_memory_context(
        self, participant_id: str, query: str = "", *, exclude_message_id: str = ""
    ) -> str:
        if not self.store or not participant_id:
            return ""
        return await self.store.memory_context(
            participant_id, query, exclude_message_id=exclude_message_id
        )

    async def latest_assistant_reply(
        self, participant_id: str, *, exclude_reply_to_id: str = ""
    ) -> str:
        if not self.store or not participant_id:
            return ""
        return await self.store.latest_assistant_reply(
            participant_id, exclude_reply_to_id=exclude_reply_to_id
        )

    async def active_news_context(self, query: str = "", *, include_unconditionally: bool = False) -> str:
        if not self.store:
            return ""
        return await self.store.active_news_context(
            query, include_unconditionally=include_unconditionally
        )

    def _message_user_id_locked(self, session_id: str) -> str | None:
        if not self._active or self._active.id != session_id:
            return None
        participant = self._participants.get(self._active.participant_token)
        return participant.id if participant else None

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
            "agent_jobs": [dict(item) for item in self._agent_jobs.values()],
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
