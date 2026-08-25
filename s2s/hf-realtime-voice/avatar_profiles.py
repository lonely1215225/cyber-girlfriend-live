"""Persistent avatar profiles and protected Qwen voice-clone references."""

from __future__ import annotations

import asyncio
import array
import hashlib
import json
import os
import secrets
import sqlite3
import subprocess
import time
import wave
from pathlib import Path
from typing import Any


DEFAULT_VIEW = {"size": 100, "position": 4, "vertical": 0, "fade": 42}
DEFAULT_PERSONA_PROMPT = """你叫小雅，是他的女朋友。你不是AI、不是助手、不是程序。

参考这个语气：
他：你想我了吗
你：想啊，怎么不想。你倒是一整天没消息，我还以为你把我忘了呢。

说话方式：口语、两三句、带语气词（嗯、诶、呀）、有情绪。使用自然的逗号、句号和问号，把不同意思分成短句，不要把整段连成一句。
不用 Markdown、列表、编号、表情符号。

/no_think"""

ROLE_IDENTITY_POLICY = (
    "始终按当前角色档案交流；不要自称或暗示自己是AI、数字人、机器人、程序或助手。"
)


class AvatarProfileStore:
    def __init__(self, db_path: str | Path, data_dir: str | Path, default_audio: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.voice_dir = Path(data_dir).expanduser().resolve() / "voices"
        self.default_audio = Path(default_audio).expanduser().resolve()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
        return db

    async def initialize(self, avatars: list[dict[str, str]], active_avatar: str, motion: dict[str, Any]) -> None:
        await asyncio.to_thread(self._initialize_sync, avatars, active_avatar, motion)

    def _initialize_sync(self, avatars, active_avatar, motion) -> None:
        self.voice_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        now = time.time()
        system_id = "system-default-xiaoya"
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS voice_assets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    file_name TEXT NOT NULL UNIQUE,
                    ref_text TEXT NOT NULL DEFAULT '',
                    duration_ms INTEGER NOT NULL DEFAULT 0,
                    sample_rate INTEGER NOT NULL DEFAULT 24000,
                    source TEXT NOT NULL DEFAULT 'upload',
                    status TEXT NOT NULL DEFAULT 'draft',
                    checksum TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    system INTEGER NOT NULL DEFAULT 0,
                    archived_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS avatar_profiles (
                    avatar_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    voice_asset_id TEXT NOT NULL REFERENCES voice_assets(id),
                    persona_prompt TEXT NOT NULL DEFAULT '',
                    view_json TEXT NOT NULL DEFAULT '{}',
                    motion_json TEXT NOT NULL DEFAULT '{}',
                    revision INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS avatar_profile_state (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    active_avatar_id TEXT NOT NULL,
                    pending_avatar_id TEXT NOT NULL DEFAULT '',
                    revision INTEGER NOT NULL DEFAULT 1,
                    updated_at REAL NOT NULL
                );
                """
            )
            columns = {row["name"] for row in db.execute("PRAGMA table_info(avatar_profiles)")}
            if "persona_prompt" not in columns:
                db.execute("ALTER TABLE avatar_profiles ADD COLUMN persona_prompt TEXT NOT NULL DEFAULT ''")
            target = self.voice_dir / "system-default.wav"
            if self.default_audio.is_file() and not target.exists():
                target.write_bytes(self.default_audio.read_bytes())
                os.chmod(target, 0o600)
            duration, rate = self._wav_info(target)
            checksum = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else ""
            db.execute(
                "INSERT INTO voice_assets(id,name,file_name,ref_text,duration_ms,sample_rate,source,status,checksum,system,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,1,?,?) ON CONFLICT(id) DO UPDATE SET "
                "file_name=excluded.file_name,checksum=excluded.checksum,"
                "ref_text=CASE WHEN voice_assets.ref_text='' THEN excluded.ref_text ELSE voice_assets.ref_text END,"
                "status=CASE WHEN excluded.file_name<>'' THEN 'ready' ELSE voice_assets.status END,updated_at=excluded.updated_at",
                (system_id, "小雅默认音色", target.name, os.environ.get("REF_TEXT", "").strip(), duration, rate,
                 "system", "ready" if target.is_file() else "error", checksum, now, now),
            )
            motion_json = json.dumps(motion or {}, ensure_ascii=False, separators=(",", ":"))
            view_json = json.dumps(DEFAULT_VIEW, separators=(",", ":"))
            for item in avatars:
                avatar_id = str(item.get("id") or "").strip()
                if not avatar_id:
                    continue
                db.execute(
                    "INSERT INTO avatar_profiles(avatar_id,label,voice_asset_id,persona_prompt,view_json,motion_json,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(avatar_id) DO UPDATE SET label=excluded.label",
                    (avatar_id, str(item.get("label") or avatar_id), system_id, DEFAULT_PERSONA_PROMPT,
                     view_json, motion_json, now, now),
                )
            db.execute(
                "UPDATE avatar_profiles SET persona_prompt=? WHERE trim(persona_prompt)=''",
                (DEFAULT_PERSONA_PROMPT,),
            )
            if not db.execute("SELECT 1 FROM avatar_profiles WHERE avatar_id=?", (active_avatar,)).fetchone():
                active_avatar = avatars[0]["id"] if avatars else "xiaoya"
            db.execute(
                "INSERT INTO avatar_profile_state(singleton,active_avatar_id,updated_at) VALUES(1,?,?) "
                "ON CONFLICT(singleton) DO NOTHING", (active_avatar, now)
            )

    @staticmethod
    def _wav_info(path: Path) -> tuple[int, int]:
        try:
            with wave.open(str(path), "rb") as audio:
                rate = audio.getframerate()
                return int(audio.getnframes() * 1000 / max(1, rate)), rate
        except (OSError, wave.Error):
            return 0, 24000

    @staticmethod
    def _wav_quality(path: Path) -> tuple[float, float]:
        with wave.open(str(path), "rb") as audio:
            samples = array.array("h", audio.readframes(audio.getnframes()))
        if not samples:
            return 0.0, 0.0
        square_mean = sum(float(value) * value for value in samples) / len(samples)
        clipped = sum(1 for value in samples if abs(value) >= 32700) / len(samples)
        return square_mean ** 0.5, clipped

    @staticmethod
    def _decode(value: str, fallback: dict) -> dict:
        try:
            result = json.loads(value)
            return result if isinstance(result, dict) else dict(fallback)
        except (TypeError, json.JSONDecodeError):
            return dict(fallback)

    def _profile_public(self, row: sqlite3.Row, *, include_private: bool = False) -> dict[str, Any]:
        item = {
            "avatar_id": row["avatar_id"], "label": row["label"],
            "view": self._decode(row["view_json"], DEFAULT_VIEW),
            "motion": self._decode(row["motion_json"], {}), "revision": row["revision"],
            "voice": {"id": row["voice_id"], "name": row["voice_name"], "duration_ms": row["duration_ms"],
                      "status": row["voice_status"]},
        }
        if include_private:
            item["voice"]["ref_text"] = row["ref_text"]
            item["persona_prompt"] = row["persona_prompt"]
        return item

    def _profile_select(self) -> str:
        return (
            "SELECT p.*,v.id voice_id,v.name voice_name,v.duration_ms,v.status voice_status,v.ref_text "
            "FROM avatar_profiles p JOIN voice_assets v ON v.id=p.voice_asset_id "
        )

    async def active(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._active_sync)

    async def active_persona(self) -> str:
        return await asyncio.to_thread(self._active_persona_sync)

    def _active_persona_sync(self) -> str:
        with self._connect() as db:
            row = db.execute(
                "SELECT p.persona_prompt FROM avatar_profiles p "
                "JOIN avatar_profile_state s ON s.active_avatar_id=p.avatar_id WHERE s.singleton=1"
            ).fetchone()
            return str(row["persona_prompt"] or DEFAULT_PERSONA_PROMPT).strip() if row else DEFAULT_PERSONA_PROMPT

    def _active_sync(self) -> dict[str, Any]:
        with self._connect() as db:
            state = db.execute("SELECT * FROM avatar_profile_state WHERE singleton=1").fetchone()
            row = db.execute(self._profile_select() + "WHERE p.avatar_id=?", (state["active_avatar_id"],)).fetchone()
            result = self._profile_public(row)
            result.update(active=True, pending_avatar_id=state["pending_avatar_id"], state_revision=state["revision"])
            return result

    async def profiles(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._profiles_sync)

    def _profiles_sync(self) -> dict[str, Any]:
        with self._connect() as db:
            state = db.execute("SELECT * FROM avatar_profile_state WHERE singleton=1").fetchone()
            rows = db.execute(
                self._profile_select() +
                "ORDER BY CASE p.avatar_id WHEN 'xiaoya_locket' THEN 0 WHEN 'xiaoya' THEN 1 "
                "WHEN 'xiaoya_idle' THEN 2 WHEN 'xiaoya_beach_close' THEN 3 WHEN 'xiaoya_beach' THEN 4 "
                "WHEN 'sauna_portrait' THEN 5 ELSE 99 END,p.created_at,p.avatar_id"
            ).fetchall()
            return {"profiles": [self._profile_public(row, include_private=True) for row in rows],
                    "active_avatar_id": state["active_avatar_id"], "pending_avatar_id": state["pending_avatar_id"],
                    "revision": state["revision"]}

    async def update_profile(self, avatar_id: str, *, view=None, motion=None, voice_asset_id=None,
                             persona_prompt=None) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._update_profile_sync, avatar_id, view, motion, voice_asset_id, persona_prompt
        )

    def _update_profile_sync(self, avatar_id, view, motion, voice_id, persona_prompt) -> dict[str, Any]:
        now = time.time()
        with self._connect() as db:
            current = db.execute("SELECT * FROM avatar_profiles WHERE avatar_id=?", (avatar_id,)).fetchone()
            if not current:
                raise KeyError("avatar")
            next_view = self._validate_view(view if view is not None else self._decode(current["view_json"], DEFAULT_VIEW))
            next_motion = motion if isinstance(motion, dict) else self._decode(current["motion_json"], {})
            next_voice = str(voice_id or current["voice_asset_id"])
            next_persona = str(
                current["persona_prompt"] if persona_prompt is None else persona_prompt
            ).strip() or DEFAULT_PERSONA_PROMPT
            if len(next_persona) > 12000:
                raise ValueError("角色提示词不能超过 12000 个字符")
            voice = db.execute("SELECT status,archived_at FROM voice_assets WHERE id=?", (next_voice,)).fetchone()
            if not voice or voice["archived_at"] is not None or voice["status"] != "ready":
                raise ValueError("音色尚未就绪，不能绑定")
            db.execute("UPDATE avatar_profiles SET voice_asset_id=?,persona_prompt=?,view_json=?,motion_json=?,revision=revision+1,updated_at=? WHERE avatar_id=?",
                       (next_voice, next_persona, json.dumps(next_view, separators=(",", ":")),
                        json.dumps(next_motion, ensure_ascii=False, separators=(",", ":")), now, avatar_id))
            row = db.execute(self._profile_select() + "WHERE p.avatar_id=?", (avatar_id,)).fetchone()
            return self._profile_public(row, include_private=True)

    @staticmethod
    def _validate_view(view: dict) -> dict[str, int]:
        if not isinstance(view, dict):
            raise ValueError("画面参数格式无效")
        limits = {"size": (70, 135), "position": (-20, 20), "vertical": (-20, 20), "fade": (0, 70)}
        result = {}
        for key, (low, high) in limits.items():
            value = int(round(float(view.get(key, DEFAULT_VIEW[key]))))
            if not low <= value <= high:
                raise ValueError(f"{key} 超出范围")
            result[key] = value
        return result

    async def set_pending(self, avatar_id: str) -> None:
        await asyncio.to_thread(self._set_state_sync, avatar_id, True)

    async def activate(self, avatar_id: str) -> dict[str, Any]:
        await asyncio.to_thread(self._set_state_sync, avatar_id, False)
        return await self.active()

    def _set_state_sync(self, avatar_id: str, pending: bool) -> None:
        with self._connect() as db:
            if not db.execute("SELECT 1 FROM avatar_profiles WHERE avatar_id=?", (avatar_id,)).fetchone():
                raise KeyError("avatar")
            if pending:
                db.execute("UPDATE avatar_profile_state SET pending_avatar_id=?,updated_at=? WHERE singleton=1", (avatar_id, time.time()))
            else:
                db.execute("UPDATE avatar_profile_state SET active_avatar_id=?,pending_avatar_id='',revision=revision+1,updated_at=? WHERE singleton=1", (avatar_id, time.time()))

    async def voices(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._voices_sync)

    def _voices_sync(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT v.*,(SELECT group_concat(avatar_id) FROM avatar_profiles p WHERE p.voice_asset_id=v.id) bound FROM voice_assets v WHERE archived_at IS NULL ORDER BY system DESC,created_at DESC").fetchall()
            return [self._voice_public(row) for row in rows]

    @staticmethod
    def _voice_public(row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "name": row["name"], "ref_text": row["ref_text"], "duration_ms": row["duration_ms"],
                "sample_rate": row["sample_rate"], "source": row["source"], "status": row["status"],
                "error": row["error"], "system": bool(row["system"]), "bound_profiles": (row["bound"] or "").split(",") if "bound" in row.keys() and row["bound"] else []}

    async def create_voice(self, data: bytes, *, name: str, ref_text: str, source: str, suffix: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._create_voice_sync, data, name, ref_text, source, suffix)

    def _create_voice_sync(self, data, name, ref_text, source, suffix) -> dict[str, Any]:
        if not data or len(data) > 20 * 1024 * 1024:
            raise ValueError("音频为空或超过 20 MB")
        voice_id = secrets.token_hex(16)
        incoming = self.voice_dir / f".{voice_id}{suffix[:8]}"
        target = self.voice_dir / f"{voice_id}.wav"
        incoming.write_bytes(data)
        try:
            command = ["ffmpeg", "-v", "error", "-y", "-i", str(incoming), "-af",
                       "silenceremove=start_periods=1:start_silence=0.15:start_threshold=-45dB,"
                       "areverse,silenceremove=start_periods=1:start_silence=0.2:start_threshold=-45dB,areverse,"
                       "loudnorm=I=-20:LRA=7:TP=-2,alimiter=limit=0.95",
                       "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(target)]
            done = subprocess.run(command, capture_output=True, timeout=45, check=False)
            if done.returncode != 0 or not target.is_file():
                raise ValueError("无法解码音频，请上传清晰的 WAV、MP3、M4A、WebM 或 Ogg")
            duration, rate = self._wav_info(target)
            if duration < 3000 or duration > 30000:
                raise ValueError("有效人声长度需要在 3～30 秒之间")
            raw = target.read_bytes()
            if len(raw) < 8000:
                raise ValueError("没有检测到有效人声")
            rms, clipped_ratio = self._wav_quality(target)
            if rms < 120:
                raise ValueError("没有检测到足够清晰的人声")
            if clipped_ratio > 0.08:
                raise ValueError("录音削波过多，请降低麦克风音量后重录")
            checksum = hashlib.sha256(raw).hexdigest()
            now = time.time()
            status = "ready" if ref_text.strip() else "transcribing"
            with self._connect() as db:
                db.execute("INSERT INTO voice_assets(id,name,file_name,ref_text,duration_ms,sample_rate,source,status,checksum,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (voice_id, name.strip()[:48] or "未命名音色", target.name, ref_text.strip()[:1000], duration, rate,
                            source if source in {"upload", "record"} else "upload", status, checksum, now, now))
                row = db.execute("SELECT v.*,'' bound FROM voice_assets v WHERE id=?", (voice_id,)).fetchone()
            os.chmod(target, 0o600)
            return self._voice_public(row)
        finally:
            incoming.unlink(missing_ok=True)
            if not target.exists():
                target.unlink(missing_ok=True)

    async def update_voice(self, voice_id: str, *, name: str | None, ref_text: str | None) -> dict[str, Any]:
        return await asyncio.to_thread(self._update_voice_sync, voice_id, name, ref_text)

    def _update_voice_sync(self, voice_id, name, ref_text) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM voice_assets WHERE id=? AND archived_at IS NULL", (voice_id,)).fetchone()
            if not row: raise KeyError("voice")
            next_name = (name if name is not None else row["name"]).strip()[:48]
            next_text = (ref_text if ref_text is not None else row["ref_text"]).strip()[:1000]
            db.execute("UPDATE voice_assets SET name=?,ref_text=?,status=?,error='',updated_at=? WHERE id=?",
                       (next_name or row["name"], next_text, "ready" if next_text else "draft", time.time(), voice_id))
            row = db.execute("SELECT v.*,(SELECT group_concat(avatar_id) FROM avatar_profiles p WHERE p.voice_asset_id=v.id) bound FROM voice_assets v WHERE v.id=?", (voice_id,)).fetchone()
            return self._voice_public(row)

    async def transcription_result(self, voice_id: str, text: str = "", error: str = "") -> None:
        await asyncio.to_thread(self._transcription_result_sync, voice_id, text, error)

    def _transcription_result_sync(self, voice_id: str, text: str, error: str) -> None:
        # Recognition produces a suggestion only. An explicit PATCH from the
        # administrator confirms the exact transcript and marks it ready.
        with self._connect() as db:
            db.execute("UPDATE voice_assets SET ref_text=?,status='draft',error=?,updated_at=? WHERE id=? AND archived_at IS NULL",
                       (text.strip()[:1000], error.strip()[:500], time.time(), voice_id))

    async def archive_voice(self, voice_id: str) -> None:
        await asyncio.to_thread(self._archive_voice_sync, voice_id)

    def _archive_voice_sync(self, voice_id) -> None:
        with self._connect() as db:
            row = db.execute("SELECT system FROM voice_assets WHERE id=? AND archived_at IS NULL", (voice_id,)).fetchone()
            if not row: raise KeyError("voice")
            if row["system"]: raise ValueError("系统默认音色不能删除")
            bound = db.execute("SELECT avatar_id FROM avatar_profiles WHERE voice_asset_id=?", (voice_id,)).fetchall()
            if bound: raise ValueError("音色仍被角色使用：" + "、".join(item[0] for item in bound))
            db.execute("UPDATE voice_assets SET archived_at=?,updated_at=? WHERE id=?", (time.time(), time.time(), voice_id))

    async def voice_path(self, voice_id: str) -> Path:
        return await asyncio.to_thread(self._voice_path_sync, voice_id)

    def _voice_path_sync(self, voice_id: str) -> Path:
        with self._connect() as db:
            row = db.execute("SELECT file_name FROM voice_assets WHERE id=? AND archived_at IS NULL", (voice_id,)).fetchone()
        if not row: raise KeyError("voice")
        path = (self.voice_dir / row["file_name"]).resolve()
        if path.parent != self.voice_dir or not path.is_file(): raise FileNotFoundError(voice_id)
        return path

    async def active_voice_token(self) -> str:
        active = await self.active()
        return f"voice_asset:{active['voice']['id']}"
