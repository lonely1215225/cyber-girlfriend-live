"""Persistent live-room identities, transcripts, admin sessions, and memory RAG."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any


_CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]{3,}")
_WORD_RE = re.compile(r"[A-Za-z0-9_]{3,}")
_FACT_PATTERNS = (
    ("name", re.compile(r"(?:我叫|叫我)([\u3400-\u9fffA-Za-z][\u3400-\u9fffA-Za-z0-9 _-]{0,23})")),
    ("like", re.compile(r"(?:我(?:很|最|比较|特别)?|也)喜欢([^，。！？!?\n]{1,32})")),
    ("dislike", re.compile(r"(?:我(?:很|最|比较|特别)?|也)不喜欢([^，。！？!?\n]{1,32})")),
    ("location", re.compile(r"我(?:住在|来自|在)([\u3400-\u9fffA-Za-z][^，。！？!?\n]{0,23})")),
    ("occupation", re.compile(r"我(?:是|做)([^，。！？!?\n]{1,24})(?:工作|的|$)")),
)


class RoomStore:
    """Small async facade over SQLite; every operation runs off the event loop."""

    def __init__(self, path: str | Path, *, message_limit: int = 80) -> None:
        self.path = Path(path).expanduser().resolve()
        self.message_limit = max(20, int(message_limit))
        self._initialized = False
        self._fts_enabled = False

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    name_zh TEXT NOT NULL DEFAULT '',
                    name_en TEXT NOT NULL DEFAULT '',
                    name_kind TEXT NOT NULL DEFAULT 'generated',
                    role TEXT NOT NULL DEFAULT 'guest',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    disabled_at REAL
                );
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    revoked_at REAL,
                    ip_address TEXT NOT NULL DEFAULT '',
                    user_agent_hash TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_user_sessions_expiry ON user_sessions(expires_at);
                CREATE TABLE IF NOT EXISTS user_ip_history (
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    ip_hash TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    first_seen_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    visit_count INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (user_id, ip_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_user_ip_last_seen ON user_ip_history(last_seen_at);
                CREATE TABLE IF NOT EXISTS call_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    granted_at REAL NOT NULL,
                    connected_at REAL,
                    ended_at REAL,
                    end_reason TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_calls_user_time ON call_sessions(user_id, granted_at DESC);
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    room_id TEXT NOT NULL DEFAULT 'default',
                    user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
                    role TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    speaker_snapshot TEXT NOT NULL,
                    content TEXT NOT NULL,
                    reply_to_id TEXT,
                    reply_to_speaker TEXT NOT NULL DEFAULT '',
                    reply_to_text TEXT NOT NULL DEFAULT '',
                    call_session_id TEXT,
                    visibility TEXT NOT NULL DEFAULT 'room',
                    interrupted INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_room_time ON chat_messages(room_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_user_time ON chat_messages(user_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS user_memories (
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    memory_key TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source_message_id TEXT,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (user_id, memory_key)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user_time ON user_memories(user_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS admin_sessions (
                    id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    revoked_at REAL,
                    ip_address TEXT NOT NULL DEFAULT '',
                    user_agent_hash TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_admin_expiry ON admin_sessions(expires_at);
                CREATE TABLE IF NOT EXISTS admin_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_session_id TEXT,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL DEFAULT '',
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    ip_address TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                """
            )
            try:
                db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chat_messages_fts "
                    "USING fts5(message_id UNINDEXED, user_id UNINDEXED, content, tokenize='trigram')"
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False
            db.execute("PRAGMA user_version=1")
        os.chmod(self.path, 0o600)
        self._initialized = True

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def _ua_hash(user_agent: str) -> str:
        return hashlib.sha256(user_agent.encode()).hexdigest() if user_agent else ""

    @staticmethod
    def _ip_hash(ip_address: str) -> str:
        return hashlib.sha256(ip_address.encode()).hexdigest() if ip_address else ""

    async def create_user(
        self,
        *,
        user_id: str,
        display_name: str,
        name_zh: str,
        name_en: str,
        token: str,
        ip_address: str = "",
        user_agent: str = "",
        ttl_seconds: int = 365 * 24 * 60 * 60,
    ) -> bool:
        return await asyncio.to_thread(
            self._create_user_sync,
            user_id,
            display_name,
            name_zh,
            name_en,
            token,
            ip_address,
            user_agent,
            ttl_seconds,
        )

    def _create_user_sync(self, user_id, display_name, name_zh, name_en, token, ip, ua, ttl) -> bool:
        now = time.time()
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO users(id,display_name,name_zh,name_en,name_kind,created_at,updated_at,last_seen_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (user_id, display_name, name_zh, name_en, "generated", now, now, now),
                )
                db.execute(
                    "INSERT INTO user_sessions(id,user_id,token_hash,created_at,last_seen_at,expires_at,ip_address,user_agent_hash) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (secrets.token_urlsafe(18), user_id, self._token_hash(token), now, now, now + ttl, ip, self._ua_hash(ua)),
                )
                self._record_ip(db, user_id, ip, now)
            return True
        except sqlite3.IntegrityError:
            return False

    def _record_ip(self, db: sqlite3.Connection, user_id: str, ip: str, now: float) -> None:
        if not ip:
            return
        db.execute(
            "INSERT INTO user_ip_history(user_id,ip_hash,ip_address,first_seen_at,last_seen_at,visit_count) "
            "VALUES(?,?,?,?,?,1) ON CONFLICT(user_id,ip_hash) DO UPDATE SET "
            "ip_address=excluded.ip_address,last_seen_at=excluded.last_seen_at,visit_count=visit_count+1",
            (user_id, self._ip_hash(ip), ip, now, now),
        )

    async def resolve_user(self, token: str, *, ip_address: str = "", user_agent: str = "") -> dict[str, Any] | None:
        return await asyncio.to_thread(self._resolve_user_sync, token, ip_address, user_agent)

    def _resolve_user_sync(self, token: str, ip: str, ua: str) -> dict[str, Any] | None:
        if not token:
            return None
        now = time.time()
        with self._connect() as db:
            row = db.execute(
                "SELECT u.* FROM user_sessions s JOIN users u ON u.id=s.user_id "
                "WHERE s.token_hash=? AND s.revoked_at IS NULL AND s.expires_at>=? AND u.disabled_at IS NULL",
                (self._token_hash(token), now),
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE user_sessions SET last_seen_at=?,ip_address=?,user_agent_hash=? WHERE token_hash=?",
                (now, ip, self._ua_hash(ua), self._token_hash(token)),
            )
            db.execute("UPDATE users SET last_seen_at=? WHERE id=?", (now, row["id"]))
            self._record_ip(db, row["id"], ip, now)
            return dict(row)

    async def rename_user(self, user_id: str, display_name: str) -> bool:
        return await asyncio.to_thread(self._rename_user_sync, user_id, display_name)

    def _rename_user_sync(self, user_id: str, display_name: str) -> bool:
        try:
            with self._connect() as db:
                cursor = db.execute(
                    "UPDATE users SET display_name=?,name_kind='custom',updated_at=? WHERE id=?",
                    (display_name, time.time(), user_id),
                )
                return cursor.rowcount == 1
        except sqlite3.IntegrityError:
            return False

    async def load_recent_messages(self, limit: int | None = None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._load_recent_messages_sync, limit or self.message_limit)

    def _load_recent_messages_sync(self, limit: int) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM chat_messages WHERE room_id='default' AND visibility='room' "
                "ORDER BY created_at DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        messages = [self._message_public(dict(row)) for row in reversed(rows)]
        return messages

    @staticmethod
    def _message_public(row: dict[str, Any]) -> dict[str, Any]:
        item = {
            "id": row["id"],
            "kind": row["kind"],
            "role": row["role"],
            "speaker": row["speaker_snapshot"],
            "text": row["content"],
            "partial": False,
            "interrupted": bool(row["interrupted"]),
            "created_at": row["created_at"],
        }
        if row.get("user_id"):
            item["participant_id"] = row["user_id"]
        if row.get("reply_to_id"):
            item["reply_to"] = {
                "id": row["reply_to_id"],
                "speaker": row.get("reply_to_speaker") or "观众",
                "text": row.get("reply_to_text") or "",
            }
        return item

    async def save_message(self, item: dict[str, Any], *, user_id: str | None = None) -> None:
        await asyncio.to_thread(self._save_message_sync, dict(item), user_id)

    def _save_message_sync(self, item: dict[str, Any], user_id: str | None) -> None:
        text = str(item.get("text") or "").strip()
        if not text or item.get("partial"):
            return
        now = time.time()
        created = float(item.get("created_at") or now)
        quote = item.get("reply_to") if isinstance(item.get("reply_to"), dict) else {}
        with self._connect() as db:
            db.execute(
                "INSERT INTO chat_messages(id,user_id,role,kind,speaker_snapshot,content,reply_to_id,"
                "reply_to_speaker,reply_to_text,call_session_id,visibility,interrupted,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "content=excluded.content,speaker_snapshot=excluded.speaker_snapshot,interrupted=excluded.interrupted,"
                "updated_at=excluded.updated_at",
                (
                    str(item.get("id") or secrets.token_urlsafe(18)), user_id,
                    str(item.get("role") or "viewer"), str(item.get("kind") or "chat"),
                    str(item.get("speaker") or "观众"), text, str(quote.get("id") or "") or None,
                    str(quote.get("speaker") or ""), str(quote.get("text") or "")[:120],
                    str(item.get("call_session_id") or "") or None,
                    str(item.get("visibility") or "room"), int(bool(item.get("interrupted"))), created, now,
                ),
            )
            if self._fts_enabled:
                db.execute("DELETE FROM chat_messages_fts WHERE message_id=?", (str(item.get("id") or ""),))
                db.execute(
                    "INSERT INTO chat_messages_fts(message_id,user_id,content) VALUES(?,?,?)",
                    (str(item.get("id") or ""), user_id or "", text),
                )
            if user_id and str(item.get("role")) in {"user", "viewer"}:
                self._extract_memories(db, user_id, str(item.get("id") or ""), text, now)

    def _extract_memories(self, db: sqlite3.Connection, user_id: str, message_id: str, text: str, now: float) -> None:
        for memory_type, pattern in _FACT_PATTERNS:
            for match in pattern.finditer(text):
                value = re.sub(r"\s+", " ", match.group(1)).strip(" ，。！？!?、")[:48]
                if not value:
                    continue
                suffix = "primary" if memory_type in {"name", "location", "occupation"} else hashlib.sha1(value.encode()).hexdigest()[:12]
                key = f"{memory_type}:{suffix}"
                db.execute(
                    "INSERT INTO user_memories(user_id,memory_key,memory_type,value,source_message_id,confidence,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,0.72,?,?) ON CONFLICT(user_id,memory_key) DO UPDATE SET "
                    "value=excluded.value,source_message_id=excluded.source_message_id,updated_at=excluded.updated_at",
                    (user_id, key, memory_type, value, message_id, now, now),
                )

    async def memory_context(self, user_id: str, query: str = "", *, max_chars: int = 3200) -> str:
        return await asyncio.to_thread(self._memory_context_sync, user_id, query, max_chars)

    @staticmethod
    def _fts_query(query: str) -> str:
        terms: list[str] = []
        for run in _CJK_RUN_RE.findall(query):
            compact = run[:24]
            terms.extend(compact[index:index + 3] for index in range(max(1, len(compact) - 2)))
        terms.extend(_WORD_RE.findall(query.lower()))
        unique = list(dict.fromkeys(term.replace('"', '') for term in terms if len(term) >= 3))[:16]
        return " OR ".join(f'"{term}"' for term in unique)

    def _memory_context_sync(self, user_id: str, query: str, max_chars: int) -> str:
        with self._connect() as db:
            user = db.execute("SELECT display_name FROM users WHERE id=?", (user_id,)).fetchone()
            if user is None:
                return ""
            facts = db.execute(
                "SELECT memory_type,value FROM user_memories WHERE user_id=? ORDER BY updated_at DESC LIMIT 12",
                (user_id,),
            ).fetchall()
            relevant: list[sqlite3.Row] = []
            fts_query = self._fts_query(query)
            if self._fts_enabled and fts_query:
                try:
                    relevant = db.execute(
                        "SELECT m.id,m.role,m.speaker_snapshot,m.content,m.created_at "
                        "FROM chat_messages_fts f JOIN chat_messages m ON m.id=f.message_id "
                        "WHERE f.user_id=? AND chat_messages_fts MATCH ? ORDER BY bm25(chat_messages_fts) LIMIT 6",
                        (user_id, fts_query),
                    ).fetchall()
                except sqlite3.OperationalError:
                    relevant = []
            recent = db.execute(
                "SELECT id,role,speaker_snapshot,content,created_at FROM chat_messages "
                "WHERE user_id=? ORDER BY created_at DESC LIMIT 10",
                (user_id,),
            ).fetchall()
        lines = [f"用户固定身份：{user['display_name']}"]
        labels = {"name": "称呼", "like": "喜欢", "dislike": "不喜欢", "location": "所在地", "occupation": "身份/职业"}
        if facts:
            lines.append("已确认的个人信息：" + "；".join(f"{labels.get(row['memory_type'], row['memory_type'])}：{row['value']}" for row in facts))
        seen: set[str] = set()
        history: list[str] = []
        for row in [*relevant, *reversed(recent)]:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            who = "用户" if row["role"] in {"user", "viewer"} else "数字人"
            history.append(f"{who}：{row['content']}")
            if len(history) >= 12:
                break
        if history:
            lines.append("仅属于该用户的历史对话：\n" + "\n".join(history))
        return "\n".join(lines)[:max_chars]

    async def start_call(self, call_id: str, user_id: str, granted_at: float, connected_at: float) -> None:
        await asyncio.to_thread(self._start_call_sync, call_id, user_id, granted_at, connected_at)

    def _start_call_sync(self, call_id, user_id, granted_at, connected_at) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO call_sessions(id,user_id,granted_at,connected_at) VALUES(?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET connected_at=excluded.connected_at",
                (call_id, user_id, granted_at, connected_at),
            )

    async def end_call(self, call_id: str, reason: str = "ended") -> None:
        await asyncio.to_thread(self._end_call_sync, call_id, reason)

    def _end_call_sync(self, call_id: str, reason: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE call_sessions SET ended_at=?,end_reason=? WHERE id=? AND ended_at IS NULL",
                (time.time(), reason[:64], call_id),
            )

    async def create_admin_session(self, *, ip_address: str, user_agent: str, ttl_seconds: int) -> str:
        token = secrets.token_urlsafe(48)
        await asyncio.to_thread(self._create_admin_session_sync, token, ip_address, user_agent, ttl_seconds)
        return token

    def _create_admin_session_sync(self, token: str, ip: str, ua: str, ttl: int) -> None:
        now = time.time()
        with self._connect() as db:
            db.execute(
                "INSERT INTO admin_sessions(id,token_hash,created_at,last_seen_at,expires_at,ip_address,user_agent_hash) "
                "VALUES(?,?,?,?,?,?,?)",
                (secrets.token_urlsafe(18), self._token_hash(token), now, now, now + ttl, ip, self._ua_hash(ua)),
            )

    async def admin_session(self, token: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._admin_session_sync, token)

    def _admin_session_sync(self, token: str) -> dict[str, Any] | None:
        if not token:
            return None
        now = time.time()
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM admin_sessions WHERE token_hash=? AND revoked_at IS NULL AND expires_at>=?",
                (self._token_hash(token), now),
            ).fetchone()
            if row:
                db.execute("UPDATE admin_sessions SET last_seen_at=? WHERE id=?", (now, row["id"]))
            return dict(row) if row else None

    async def revoke_admin_session(self, token: str) -> None:
        await asyncio.to_thread(self._revoke_admin_session_sync, token)

    def _revoke_admin_session_sync(self, token: str) -> None:
        if not token:
            return
        with self._connect() as db:
            db.execute(
                "UPDATE admin_sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL",
                (time.time(), self._token_hash(token)),
            )

    async def audit_admin(self, session_id: str, action: str, target: str, detail_json: str, ip_address: str) -> None:
        await asyncio.to_thread(self._audit_admin_sync, session_id, action, target, detail_json, ip_address)

    def _audit_admin_sync(self, session_id, action, target, detail_json, ip) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO admin_audit_logs(admin_session_id,action,target,detail_json,ip_address,created_at) VALUES(?,?,?,?,?,?)",
                (session_id, action[:80], target[:120], detail_json[:4000], ip, time.time()),
            )

    async def cleanup(self, *, ip_retention_days: int = 30) -> None:
        await asyncio.to_thread(self._cleanup_sync, ip_retention_days)

    def _cleanup_sync(self, ip_retention_days: int) -> None:
        now = time.time()
        cutoff = now - max(1, ip_retention_days) * 86400
        with self._connect() as db:
            db.execute("DELETE FROM user_ip_history WHERE last_seen_at<?", (cutoff,))
            db.execute("UPDATE user_sessions SET ip_address='' WHERE last_seen_at<?", (cutoff,))
            db.execute("DELETE FROM user_sessions WHERE expires_at<? OR revoked_at IS NOT NULL", (now - 86400,))
            db.execute("DELETE FROM admin_sessions WHERE expires_at<? OR revoked_at IS NOT NULL", (now - 86400,))
