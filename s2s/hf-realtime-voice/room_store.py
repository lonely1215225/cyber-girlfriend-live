"""Persistent live-room identities, transcripts, admin sessions, and memory RAG."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
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
                CREATE TABLE IF NOT EXISTS agent_jobs (
                    id TEXT PRIMARY KEY,
                    room_id TEXT NOT NULL DEFAULT 'default',
                    message_id TEXT NOT NULL,
                    participant_id TEXT NOT NULL DEFAULT '',
                    speaker TEXT NOT NULL DEFAULT '',
                    prompt TEXT NOT NULL,
                    phase TEXT NOT NULL DEFAULT 'queued',
                    status_text TEXT NOT NULL DEFAULT '',
                    route_json TEXT NOT NULL DEFAULT '{}',
                    evidence_json TEXT NOT NULL DEFAULT '{}',
                    task_spec_json TEXT NOT NULL DEFAULT '{}',
                    tool_plan_json TEXT NOT NULL DEFAULT '{}',
                    evidence_items_json TEXT NOT NULL DEFAULT '[]',
                    answer_draft_json TEXT NOT NULL DEFAULT '{}',
                    coverage REAL NOT NULL DEFAULT 0,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    candidate_text TEXT NOT NULL DEFAULT '',
                    validation_json TEXT NOT NULL DEFAULT '[]',
                    final_text TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    feedback_count INTEGER NOT NULL DEFAULT 0,
                    terminal INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(room_id, message_id)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_jobs_room_time ON agent_jobs(room_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_agent_jobs_participant ON agent_jobs(participant_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_agent_jobs_terminal ON agent_jobs(terminal, updated_at);
                CREATE TABLE IF NOT EXISTS conversation_focus (
                    participant_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL DEFAULT '',
                    subject_label TEXT NOT NULL DEFAULT '',
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    last_intent TEXT NOT NULL DEFAULT '',
                    last_question TEXT NOT NULL DEFAULT '',
                    last_answer TEXT NOT NULL DEFAULT '',
                    evidence_time TEXT NOT NULL DEFAULT '',
                    expires_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_focus_expiry ON conversation_focus(expires_at);
                CREATE TABLE IF NOT EXISTS active_news_topics (
                    room_id TEXT PRIMARY KEY,
                    topic_id TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '新闻',
                    title TEXT NOT NULL,
                    title_normalized TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    source_url TEXT NOT NULL DEFAULT '',
                    published_at TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '',
                    broadcast_text TEXT NOT NULL DEFAULT '',
                    message_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'selected',
                    locked_until REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS news_broadcast_fingerprints (
                    fingerprint TEXT PRIMARY KEY,
                    title_normalized TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    canonical_url_hash TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    broadcasted_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_news_fingerprints_time
                    ON news_broadcast_fingerprints(broadcasted_at DESC);
                CREATE TABLE IF NOT EXISTS admin_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_session_id TEXT,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL DEFAULT '',
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    ip_address TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS room_settings (
                    setting_key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    updated_at REAL NOT NULL
                );
                """
            )
            agent_columns = {row["name"] for row in db.execute("PRAGMA table_info(agent_jobs)")}
            agent_migrations = {
                "metrics_json": "TEXT NOT NULL DEFAULT '{}'",
                "task_spec_json": "TEXT NOT NULL DEFAULT '{}'",
                "tool_plan_json": "TEXT NOT NULL DEFAULT '{}'",
                "evidence_items_json": "TEXT NOT NULL DEFAULT '[]'",
                "answer_draft_json": "TEXT NOT NULL DEFAULT '{}'",
                "coverage": "REAL NOT NULL DEFAULT 0",
                "retry_count": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, definition in agent_migrations.items():
                if column not in agent_columns:
                    db.execute(f"ALTER TABLE agent_jobs ADD COLUMN {column} {definition}")  # noqa: S608
            try:
                db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chat_messages_fts "
                    "USING fts5(message_id UNINDEXED, user_id UNINDEXED, content, tokenize='trigram')"
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False
            # Welcome IDs have always contained the target participant. Older
            # versions saved those public assistant messages without user_id,
            # so backfill the owner once and make existing welcomes available
            # to that participant's private conversational context.
            db.execute(
                "UPDATE chat_messages SET user_id=("
                "SELECT users.id FROM users "
                "WHERE chat_messages.id LIKE 'bot:welcome:' || users.id || ':%' LIMIT 1"
                ") WHERE user_id IS NULL AND id LIKE 'bot:welcome:%'"
            )
            if self._fts_enabled:
                db.execute(
                    "UPDATE chat_messages_fts SET user_id=COALESCE(("
                    "SELECT chat_messages.user_id FROM chat_messages "
                    "WHERE chat_messages.id=chat_messages_fts.message_id"
                    "),'') WHERE message_id LIKE 'bot:welcome:%'"
                )
            db.execute("PRAGMA user_version=1")
        os.chmod(self.path, 0o600)
        self._initialized = True

    async def room_setting(self, key: str, default: Any = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._room_setting_sync, key, default)

    def _room_setting_sync(self, key: str, default: Any) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT value_json,revision,updated_at FROM room_settings WHERE setting_key=?",
                (key,),
            ).fetchone()
        if row is None:
            return {"key": key, "value": default, "revision": 0, "updated_at": 0.0}
        try:
            value = json.loads(row["value_json"])
        except (TypeError, json.JSONDecodeError):
            value = default
        return {
            "key": key,
            "value": value,
            "revision": int(row["revision"] or 0),
            "updated_at": float(row["updated_at"] or 0.0),
        }

    async def set_room_setting(self, key: str, value: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._set_room_setting_sync, key, value)

    def _set_room_setting_sync(self, key: str, value: Any) -> dict[str, Any]:
        now = time.time()
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as db:
            db.execute(
                "INSERT INTO room_settings(setting_key,value_json,revision,updated_at) "
                "VALUES(?,?,1,?) ON CONFLICT(setting_key) DO UPDATE SET "
                "value_json=excluded.value_json,revision=room_settings.revision+1,"
                "updated_at=excluded.updated_at",
                (key, encoded, now),
            )
            row = db.execute(
                "SELECT revision,updated_at FROM room_settings WHERE setting_key=?",
                (key,),
            ).fetchone()
        return {
            "key": key,
            "value": value,
            "revision": int(row["revision"]),
            "updated_at": float(row["updated_at"]),
        }

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

    async def save_agent_job(self, job: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._save_agent_job_sync, dict(job))

    def _save_agent_job_sync(self, job: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        job_id = str(job.get("id") or secrets.token_urlsafe(18))
        with self._connect() as db:
            db.execute(
                "INSERT INTO agent_jobs(id,room_id,message_id,participant_id,speaker,prompt,phase,status_text,"
                "route_json,evidence_json,task_spec_json,tool_plan_json,evidence_items_json,answer_draft_json,coverage,retry_count,"
                "candidate_text,validation_json,final_text,error,metrics_json,feedback_count,terminal,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(room_id,message_id) DO UPDATE SET "
                "phase=excluded.phase,status_text=excluded.status_text,route_json=excluded.route_json,"
                "evidence_json=excluded.evidence_json,candidate_text=excluded.candidate_text,"
                "task_spec_json=excluded.task_spec_json,tool_plan_json=excluded.tool_plan_json,"
                "evidence_items_json=excluded.evidence_items_json,answer_draft_json=excluded.answer_draft_json,"
                "coverage=excluded.coverage,retry_count=excluded.retry_count,"
                "validation_json=excluded.validation_json,final_text=excluded.final_text,error=excluded.error,"
                "metrics_json=excluded.metrics_json,"
                "feedback_count=excluded.feedback_count,terminal=excluded.terminal,updated_at=excluded.updated_at",
                (
                    job_id, str(job.get("room_id") or "default"), str(job.get("message_id") or ""),
                    str(job.get("participant_id") or ""), str(job.get("speaker") or ""),
                    str(job.get("prompt") or ""), str(job.get("phase") or "queued"),
                    str(job.get("status_text") or ""),
                    json.dumps(job.get("route") or {}, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(job.get("evidence") or {}, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(job.get("task_spec") or {}, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(job.get("tool_plan") or {}, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(job.get("evidence_items") or [], ensure_ascii=False, separators=(",", ":")),
                    json.dumps(job.get("answer_draft") or {}, ensure_ascii=False, separators=(",", ":")),
                    max(0.0, min(1.0, float(job.get("coverage") or 0))),
                    max(0, int(job.get("retry_count") or 0)),
                    str(job.get("candidate_text") or ""),
                    json.dumps(job.get("validation_errors") or [], ensure_ascii=False, separators=(",", ":")),
                    str(job.get("final_text") or ""), str(job.get("error") or "")[:1000],
                    json.dumps(job.get("metrics") or {}, ensure_ascii=False, separators=(",", ":")),
                    max(0, int(job.get("feedback_count") or 0)), int(bool(job.get("terminal"))),
                    float(job.get("created_at") or now), now,
                ),
            )
            row = db.execute("SELECT * FROM agent_jobs WHERE room_id=? AND message_id=?", (
                str(job.get("room_id") or "default"), str(job.get("message_id") or ""),
            )).fetchone()
        return self._agent_job_public(dict(row))

    @staticmethod
    def _agent_job_public(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"], "message_id": row["message_id"],
            "participant_id": row.get("participant_id") or "", "speaker": row.get("speaker") or "",
            "prompt": row.get("prompt") or "", "phase": row.get("phase") or "queued",
            "status_text": row.get("status_text") or "", "final_text": row.get("final_text") or "",
            "feedback_count": int(row.get("feedback_count") or 0),
            "terminal": bool(row.get("terminal")), "created_at": row.get("created_at") or 0,
            "updated_at": row.get("updated_at") or 0,
        }

    async def load_agent_jobs(self, *, recoverable_only: bool = False, limit: int = 30) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._load_agent_jobs_sync, recoverable_only, limit)

    def _load_agent_jobs_sync(self, recoverable_only: bool, limit: int) -> list[dict[str, Any]]:
        where = "WHERE room_id='default'" + (" AND terminal=0" if recoverable_only else "")
        with self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM agent_jobs {where} ORDER BY updated_at DESC LIMIT ?",  # noqa: S608
                (max(1, min(100, int(limit))),),
            ).fetchall()
        return [self._agent_job_public(dict(row)) for row in reversed(rows)]

    async def recent_agent_context(self, participant_id: str, *, max_chars: int = 800) -> str:
        return await asyncio.to_thread(self._recent_agent_context_sync, participant_id, max_chars)

    def _recent_agent_context_sync(self, participant_id: str, max_chars: int) -> str:
        if not participant_id:
            return ""
        with self._connect() as db:
            rows = db.execute(
                "SELECT prompt,final_text FROM agent_jobs WHERE participant_id=? AND terminal=1 "
                "ORDER BY updated_at DESC LIMIT 3", (participant_id,),
            ).fetchall()
        lines = [f"问题：{row['prompt']}\n结果：{row['final_text']}" for row in reversed(rows) if row["final_text"]]
        return "\n".join(lines)[-max(100, int(max_chars)):]

    async def get_conversation_focus(self, participant_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_conversation_focus_sync, participant_id)

    def _get_conversation_focus_sync(self, participant_id: str) -> dict[str, Any]:
        if not participant_id:
            return {}
        now = time.time()
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM conversation_focus WHERE participant_id=? AND expires_at>?",
                (participant_id, now),
            ).fetchone()
        if not row:
            return {}
        item = dict(row)
        with contextlib.suppress(json.JSONDecodeError):
            item["aliases"] = json.loads(item.pop("aliases_json", "[]"))
        return item

    async def set_conversation_focus(
        self, participant_id: str, task_spec: dict[str, Any], answer: str, evidence_time: str = ""
    ) -> None:
        await asyncio.to_thread(
            self._set_conversation_focus_sync, participant_id, dict(task_spec), answer, evidence_time
        )

    def _set_conversation_focus_sync(
        self, participant_id: str, task_spec: dict[str, Any], answer: str, evidence_time: str
    ) -> None:
        if not participant_id or not task_spec.get("subject"):
            return
        now = time.time()
        ttl = max(300, int(os.environ.get("AGENT_FOCUS_TTL_SECONDS", "1800")))
        with self._connect() as db:
            db.execute(
                "INSERT INTO conversation_focus(participant_id,subject,subject_label,aliases_json,last_intent,"
                "last_question,last_answer,evidence_time,expires_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(participant_id) DO UPDATE SET subject=excluded.subject,subject_label=excluded.subject_label,"
                "aliases_json=excluded.aliases_json,last_intent=excluded.last_intent,last_question=excluded.last_question,"
                "last_answer=excluded.last_answer,evidence_time=excluded.evidence_time,expires_at=excluded.expires_at,"
                "updated_at=excluded.updated_at",
                (
                    participant_id, str(task_spec.get("subject") or "")[:100],
                    str(task_spec.get("subject_label") or "")[:100],
                    json.dumps(task_spec.get("aliases") or [], ensure_ascii=False, separators=(",", ":")),
                    str(task_spec.get("intent") or "")[:40], str(task_spec.get("resolved_question") or "")[:500],
                    str(answer or "")[:1200], str(evidence_time or "")[:80], now + ttl, now,
                ),
            )

    async def recent_news_titles(self, *, days: int = 7, limit: int = 500) -> list[str]:
        return await asyncio.to_thread(self._recent_news_titles_sync, days, limit)

    def _recent_news_titles_sync(self, days: int, limit: int) -> list[str]:
        cutoff = time.time() - max(1, int(days)) * 86400
        with self._connect() as db:
            rows = db.execute(
                "SELECT title FROM news_broadcast_fingerprints WHERE broadcasted_at>=? "
                "ORDER BY broadcasted_at DESC LIMIT ?",
                (cutoff, max(1, min(1000, int(limit)))),
            ).fetchall()
        return [str(row["title"] or "") for row in rows if row["title"]]

    async def set_active_news_topic(self, topic: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._set_active_news_topic_sync, dict(topic))

    async def can_replace_active_news(self, room_id: str = "default") -> bool:
        return await asyncio.to_thread(self._can_replace_active_news_sync, room_id)

    def _can_replace_active_news_sync(self, room_id: str) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT status,locked_until FROM active_news_topics WHERE room_id=?", (room_id,)
            ).fetchone()
        return row is None or row["status"] != "discussed" or float(row["locked_until"] or 0) < time.time()

    def _set_active_news_topic_sync(self, topic: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        room_id = str(topic.get("room_id") or "default")
        title = str(topic.get("title") or "")[:300]
        normalized = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", str(topic.get("title_normalized") or title).lower())
        fingerprint = str(topic.get("fingerprint") or hashlib.sha256(normalized.encode()).hexdigest()[:32])
        topic_id = str(topic.get("topic_id") or f"news_{fingerprint[:24]}")
        with self._connect() as db:
            # One complete topic per room: this UPSERT atomically replaces the
            # prior subject instead of accumulating RSS bodies indefinitely.
            db.execute(
                "INSERT INTO active_news_topics(room_id,topic_id,category,title,title_normalized,summary,source,"
                "source_url,published_at,evidence,broadcast_text,message_id,status,locked_until,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(room_id) DO UPDATE SET "
                "topic_id=excluded.topic_id,category=excluded.category,title=excluded.title,"
                "title_normalized=excluded.title_normalized,summary=excluded.summary,source=excluded.source,"
                "source_url=excluded.source_url,published_at=excluded.published_at,evidence=excluded.evidence,"
                "broadcast_text=excluded.broadcast_text,message_id=excluded.message_id,status=excluded.status,"
                "locked_until=excluded.locked_until,created_at=excluded.created_at,updated_at=excluded.updated_at",
                (
                    room_id, topic_id, str(topic.get("category") or "新闻")[:20], title, normalized,
                    str(topic.get("summary") or "")[:1200], str(topic.get("source") or "")[:100],
                    str(topic.get("source_url") or "")[:800], str(topic.get("published_at") or "")[:80],
                    str(topic.get("evidence") or "")[:3000], str(topic.get("broadcast_text") or "")[:2000],
                    str(topic.get("message_id") or "")[:160], str(topic.get("status") or "selected")[:24],
                    float(topic.get("locked_until") or now + 900), now, now,
                ),
            )
            db.execute(
                "INSERT INTO news_broadcast_fingerprints(fingerprint,title_normalized,title,canonical_url_hash,source,broadcasted_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(fingerprint) DO UPDATE SET broadcasted_at=excluded.broadcasted_at",
                (
                    fingerprint, normalized, title, str(topic.get("canonical_url_hash") or "")[:64],
                    str(topic.get("source") or "")[:100], now,
                ),
            )
            # Hard cap as well as age retention, keeping storage bounded even
            # when a deployment broadcasts unusually frequently.
            db.execute("DELETE FROM news_broadcast_fingerprints WHERE broadcasted_at<?", (now - 7 * 86400,))
            db.execute(
                "DELETE FROM news_broadcast_fingerprints WHERE fingerprint NOT IN "
                "(SELECT fingerprint FROM news_broadcast_fingerprints ORDER BY broadcasted_at DESC LIMIT 500)"
            )
            row = db.execute("SELECT * FROM active_news_topics WHERE room_id=?", (room_id,)).fetchone()
        return dict(row)

    async def finalize_active_news_broadcast(self, text: str, message_id: str) -> None:
        await asyncio.to_thread(self._finalize_active_news_broadcast_sync, text, message_id)

    def _finalize_active_news_broadcast_sync(self, text: str, message_id: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE active_news_topics SET broadcast_text=?,message_id=?,status='broadcast',updated_at=? "
                "WHERE room_id='default'",
                (str(text or "")[:2000], str(message_id or "")[:160], time.time()),
            )

    async def active_news_context(self, query: str = "", *, max_chars: int = 1800,
                                  include_unconditionally: bool = False) -> str:
        return await asyncio.to_thread(
            self._active_news_context_sync, query, max_chars, include_unconditionally
        )

    def _active_news_context_sync(self, query: str, max_chars: int,
                                  include_unconditionally: bool) -> str:
        with self._connect() as db:
            row = db.execute("SELECT * FROM active_news_topics WHERE room_id='default'").fetchone()
        if row is None:
            return ""
        item = dict(row)
        # A current topic remains available briefly for deictic follow-ups. For
        # explicit entity/title matches it can still be used after the soft TTL.
        clean_query = re.sub(r"[\s，。！？,.!?@]", "", query or "").lower()
        deictic = bool(re.search(r"这个|这条|刚才|刚刚|那个|它|此事|这件事|是真的吗|为什么|后来|进展", clean_query))
        title = str(item.get("title") or "").lower()
        terms = [token for token in re.findall(r"[a-z0-9]{3,}|[\u3400-\u9fff]{2,}", clean_query) if token not in {"什么", "怎么", "新闻", "现在", "最新"}]
        matched = any(term in title or term in str(item.get("evidence") or "").lower() for term in terms)
        fresh = float(item.get("locked_until") or 0) >= time.time()
        if include_unconditionally and not fresh:
            return ""
        if not include_unconditionally and not matched and not (fresh and deictic):
            return ""
        if not include_unconditionally:
            # A real follow-up pins the subject briefly so the scheduler cannot
            # replace it with the next headline midway through discussion.
            with self._connect() as db:
                db.execute(
                    "UPDATE active_news_topics SET status='discussed',locked_until=?,updated_at=? "
                    "WHERE room_id='default' AND topic_id=?",
                    (time.time() + 5 * 60, time.time(), item["topic_id"]),
                )
        lines = [
            "【直播间当前新闻话题】",
            f"标题：{item['title']}",
            f"分类/来源：{item['category']} / {item['source'] or '未标注'}",
        ]
        if item.get("published_at"):
            lines.append(f"发布时间：{item['published_at']}")
        if item.get("summary"):
            lines.append(f"摘要：{item['summary']}")
        if item.get("evidence"):
            lines.append(f"播报依据：{item['evidence']}")
        if item.get("broadcast_text"):
            lines.append(f"数字人刚才实际说的是：{item['broadcast_text']}")
        lines.append("仅当用户明显在追问这条新闻时使用；若询问‘现在/最新/后来’，必须重新查询，不能把旧资料当实时结果。")
        return "\n".join(lines)[:max(300, int(max_chars))]

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
                        "LEFT JOIN agent_jobs j ON j.room_id=m.room_id AND j.message_id=m.reply_to_id "
                        "WHERE f.user_id=? AND chat_messages_fts MATCH ? "
                        "AND NOT (m.role='assistant' AND COALESCE(j.terminal,0)=1 "
                        "AND COALESCE(j.phase,'')<>'completed') "
                        "ORDER BY bm25(chat_messages_fts) LIMIT 6",
                        (user_id, fts_query),
                    ).fetchall()
                except sqlite3.OperationalError:
                    relevant = []
            recent = db.execute(
                "SELECT m.id,m.role,m.speaker_snapshot,m.content,m.created_at FROM chat_messages m "
                "LEFT JOIN agent_jobs j ON j.room_id=m.room_id AND j.message_id=m.reply_to_id "
                "WHERE m.user_id=? "
                "AND NOT (m.role='assistant' AND COALESCE(j.terminal,0)=1 "
                "AND COALESCE(j.phase,'')<>'completed') "
                "ORDER BY m.created_at DESC LIMIT 10",
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
            db.execute("DELETE FROM agent_jobs WHERE terminal=1 AND updated_at<?", (now - 7 * 86400,))
            db.execute("DELETE FROM conversation_focus WHERE expires_at<?", (now,))
            db.execute("DELETE FROM news_broadcast_fingerprints WHERE broadcasted_at<?", (now - 7 * 86400,))
