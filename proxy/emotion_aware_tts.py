"""Stable, emotion-aware Qwen3-TTS voice cloning with one reference audio."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from sensevoice_stt import SenseVoiceMetadata, get_turn_context
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler

LOG = logging.getLogger(__name__)

_STYLE_INSTRUCTIONS = {
    "neutral": "像和熟人面对面聊天一样自然地说。按语义短句呼吸和停顿，有轻重音与语调变化，不要一口气念完，也不要像播报或朗诵。",
    "gentle": "用温柔、关心、真诚的语气说，语速稍慢。按语义短句轻轻停顿，重点柔和，像在认真安慰亲近的人，不要一口气念完。",
    "calm": "用平静、耐心、不对抗的语气说。声音柔和稳定，按意思自然停顿和强调，让对方感到被理解，不要机械播报。",
    "cheerful": "用明亮、亲切、带一点笑意的语气说。节奏轻快但有呼吸和停顿，重点词稍微上扬，不要夸张或一口气念完。",
    "serious": "用自然、清楚的聊天语气说。事实和重点之间自然停顿，不要压低音调，不要像新闻播音，也不要把整段连成一句。",
}
_VOICE_IDENTITY_LOCK = (
    "始终保持参考音频中同一个年轻女性的音色、音域和说话人身份；"
    "情绪只改变语气和节奏，不要改变声线，不要变成低沉或男性音色。"
)

_GENTLE_WORDS = (
    "别怕",
    "没事",
    "我在",
    "抱抱",
    "辛苦",
    "难过",
    "委屈",
    "慢慢",
    "陪你",
    "心疼",
)
_CHEERFUL_WORDS = ("欢迎", "开心", "太好了", "恭喜", "哈哈", "嘿", "哇", "真棒", "好耶")
_SERIOUS_WORDS = (
    "新闻",
    "报道",
    "风险",
    "提醒",
    "目前",
    "数据显示",
    "需要注意",
    "局势",
    "发生",
)
_SPEECHABLE_WITH_CJK_PUNCTUATION = re.compile(
    r"[^\w\s.,!?;:'\"\-()\/\\@#%&*+=$€£¥₹₽¢\[\]{}<>~`^|…—–，。！？；：、（）【】《》“”‘’￥]",
    flags=re.UNICODE,
)


def remove_unspeechable_preserving_cjk(text: str) -> str:
    """Remove emoji/control glyphs without deleting the model's CJK punctuation."""

    return _SPEECHABLE_WITH_CJK_PUNCTUATION.sub("", text)


def split_streaming_sentences(text: str) -> list[str]:
    """Split completed Chinese sentences while retaining the unfinished tail.

    The upstream Responses handler uses NLTK's English Punkt tokenizer. It
    treats an entire Chinese reply containing ``。！？`` as one sentence, which
    makes its nominal LLM stream wait until completion before feeding TTS.
    A trailing empty item is intentional: the upstream loop then emits a
    sentence immediately when its closing punctuation arrives.
    """
    if re.search(r"[\u3400-\u9fff]", text):
        return re.split(r"(?<=[。！？!?])", text)
    from nltk import sent_tokenize as nltk_sent_tokenize

    return nltk_sent_tokenize(text)


def choose_tts_style(text: str, metadata: SenseVoiceMetadata | None = None) -> str:
    """Choose the reply delivery style from user acoustics and reply semantics."""

    if metadata:
        if metadata.emotion in {"难过", "害怕", "厌恶"} or metadata.event == "哭声":
            return "gentle"
        if metadata.emotion == "生气":
            return "calm"
        if metadata.emotion in {"开心", "惊讶"} or metadata.event in {"笑声", "掌声"}:
            return "cheerful"

    compact = re.sub(r"\s+", "", text)
    if any(word in compact for word in _GENTLE_WORDS):
        return "gentle"
    if any(word in compact for word in _CHEERFUL_WORDS):
        return "cheerful"
    if any(word in compact for word in _SERIOUS_WORDS):
        return "serious"
    return "neutral"


def prepare_tts_text(
    text: str,
    style: str,
    *,
    prosody_enabled: bool = True,
    max_clause_chars: int = 20,
) -> str:
    """Normalize markup while preserving the model's words and punctuation."""

    value = text.strip()
    value = re.sub(r"\.{3,}", "……", value)
    value = re.sub(r"…{3,}", "……", value)
    value = re.sub(r"[*_`#]+", "", value)
    value = re.sub(
        r"(?<=[\u3400-\u9fff，。！？；：……])\s+(?=[\u3400-\u9fff，。！？；：……])",
        "",
        value,
    )
    value = re.sub(r"[ \t]+", " ", value).strip()
    return value


class EmotionAwareQwen3TTSHandler(Qwen3TTSHandler):
    """Qwen3 handler with bounded sampling and per-turn acoustic guidance."""

    def setup(self, *args: Any, **kwargs: Any) -> None:
        self.emotion_control_enabled = os.environ.get("TTS_EMOTION_ENABLED", "1") != "0"
        self.style_instruct_enabled = (
            os.environ.get("TTS_STYLE_INSTRUCT_ENABLED", "1") != "0"
        )
        self.prosody_enabled = os.environ.get("TTS_PROSODY_ENABLED", "1") != "0"
        self.prosody_max_clause_chars = max(
            12, int(os.environ.get("TTS_PROSODY_MAX_CLAUSE_CHARS", "20"))
        )
        self.tts_temperature = float(os.environ.get("TTS_TEMPERATURE", "0.65"))
        self.tts_top_k = int(os.environ.get("TTS_TOP_K", "30"))
        self.tts_top_p = float(os.environ.get("TTS_TOP_P", "0.85"))
        self.tts_repetition_penalty = float(
            os.environ.get("TTS_REPETITION_PENALTY", "1.05")
        )
        self.tts_do_sample = os.environ.get("TTS_DO_SAMPLE", "1") != "0"
        self._active_style = "neutral"
        super().setup(*args, **kwargs)
        LOG.info(
            "Emotion-aware TTS enabled=%s instruct=%s temperature=%.2f top_k=%d top_p=%.2f",
            self.emotion_control_enabled,
            self.style_instruct_enabled,
            self.tts_temperature,
            self.tts_top_k,
            self.tts_top_p,
        )

    def _coalesce_pending_tts_input(
        self, current_input: TTSInput
    ) -> tuple[str, str | None, bool]:
        text, language_code, saw_end = super()._coalesce_pending_tts_input(
            current_input
        )
        metadata = get_turn_context(current_input.turn_id, current_input.turn_revision)
        self._active_style = (
            choose_tts_style(text, metadata)
            if self.emotion_control_enabled
            else "neutral"
        )
        spoken_text = prepare_tts_text(
            text,
            self._active_style,
            prosody_enabled=self.prosody_enabled,
            max_clause_chars=self.prosody_max_clause_chars,
        )
        LOG.info(
            "TTS delivery style=%s user_emotion=%s event=%s text=%s",
            self._active_style,
            metadata.emotion if metadata else None,
            metadata.event if metadata else None,
            spoken_text,
        )
        return spoken_text, language_code, saw_end

    def _apply_session_voice_override(self, model_type, runtime_config=None, response=None) -> None:
        """Resolve opaque profile voice ids without exposing server paths."""
        session_voice = None
        if response and response.audio and response.audio.output:
            session_voice = str(response.audio.output.voice or "")
        if not session_voice and runtime_config is not None:
            audio = runtime_config.session.audio
            output = audio.output if audio is not None else None
            session_voice = str(output.voice or "") if output is not None else ""
        if session_voice == "active_profile" or session_voice.startswith("voice_asset:"):
            voice_id = session_voice.partition(":")[2] if ":" in session_voice else ""
            db_path = Path(os.environ.get("ROOM_DB_PATH", "/root/cyber-girlfriend/data/live_room.sqlite3")).resolve()
            try:
                with sqlite3.connect(db_path, timeout=2) as db:
                    db.row_factory = sqlite3.Row
                    if not voice_id:
                        row = db.execute(
                            "SELECT v.* FROM avatar_profile_state s JOIN avatar_profiles p ON p.avatar_id=s.active_avatar_id "
                            "JOIN voice_assets v ON v.id=p.voice_asset_id WHERE s.singleton=1 AND v.archived_at IS NULL"
                        ).fetchone()
                    else:
                        row = db.execute(
                            "SELECT * FROM voice_assets WHERE id=? AND archived_at IS NULL AND status='ready'", (voice_id,)
                        ).fetchone()
                if not row:
                    raise ValueError("voice asset is not ready")
                voice_dir = Path(os.environ.get(
                    "VOICE_ASSET_DIR", str(Path(__file__).resolve().parents[1] / "data" / "voices")
                )).resolve()
                audio_path = (voice_dir / row["file_name"]).resolve()
                if audio_path.parent != voice_dir.resolve() or not audio_path.is_file():
                    raise ValueError("voice asset audio is missing")
                self.ref_audio = str(audio_path)
                self.ref_text = str(row["ref_text"] or "")
                LOG.info("Using protected voice asset %s (%s)", row["id"], row["name"])
                return
            except (sqlite3.Error, OSError, ValueError) as exc:
                LOG.warning("Ignoring unavailable profile voice %r: %s", session_voice, exc)
                return
        super()._apply_session_voice_override(model_type, runtime_config, response)

    def _process_voice_clone(self, text: str) -> Iterator[bytes | np.ndarray]:
        if self.backend != "faster_qwen3_tts":
            yield from super()._process_voice_clone(text)
            return

        max_new_tokens = self._estimate_max_new_tokens(text)
        instruct = None
        if self.style_instruct_enabled:
            style = _STYLE_INSTRUCTIONS.get(self._active_style, _STYLE_INSTRUCTIONS["neutral"])
            instruct = f"{style}{_VOICE_IDENTITY_LOCK}"

        yield from self._stream(
            self.model.generate_voice_clone_streaming(
                text=text,
                language=self.language,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                xvec_only=self.xvec_only,
                chunk_size=self.streaming_chunk_size,
                max_new_tokens=max_new_tokens,
                parity_mode=self.parity_mode,
                non_streaming_mode=self.non_streaming_mode,
                instruct=instruct,
                temperature=self.tts_temperature,
                top_k=self.tts_top_k,
                top_p=self.tts_top_p,
                do_sample=self.tts_do_sample,
                repetition_penalty=self.tts_repetition_penalty,
            ),
            label=f"voice_clone_{self._active_style}",
        )


def install_emotion_aware_tts() -> None:
    import speech_to_speech.LLM.responses_api_language_model as responses_module
    import speech_to_speech.LLM.utils as llm_utils_module
    import speech_to_speech.TTS.qwen3_tts_handler as handler_module

    handler_module.Qwen3TTSHandler = EmotionAwareQwen3TTSHandler
    llm_utils_module.remove_unspeechable = remove_unspeechable_preserving_cjk
    # responses_api_language_model imports the helper into its module scope,
    # so patch that reference too. This fixes the actual punctuation loss
    # without guessing or rewriting any model output.
    responses_module.remove_unspeechable = remove_unspeechable_preserving_cjk
    responses_module.sent_tokenize = split_streaming_sentences
    LOG.info("Emotion-aware Qwen3-TTS and CJK punctuation preservation installed")
