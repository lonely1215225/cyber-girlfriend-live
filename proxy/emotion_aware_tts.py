"""Stable, emotion-aware Qwen3-TTS voice cloning with one reference audio."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from sensevoice_stt import SenseVoiceMetadata, get_turn_context
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler

from expression_director import (
    DELIVERY_CONTROL_PROMPT,
    DeliveryControlFilter,
    publish_expression,
)

# Reuse the handler logger, which the service explicitly keeps at INFO. The
# module's former top-level logger stayed at WARNING, hiding the exact text and
# pause hints that reached TTS and making punctuation regressions hard to audit.
LOG = logging.getLogger("speech_to_speech.TTS.qwen3_tts_handler")

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

_SPEECHABLE_WITH_CJK_PUNCTUATION = re.compile(
    r"[^\w\s.,!?;:'\"\-()\/\\@#%&*+=$€£¥₹₽¢\[\]{}<>~`^|…—–，。！？；：、（）【】《》“”‘’￥]",
    flags=re.UNICODE,
)

_delivery_filter_local = threading.local()


def remove_unspeechable_preserving_cjk(text: str) -> str:
    """Hide LLM delivery controls and preserve the model's CJK punctuation."""

    delivery_filter = getattr(_delivery_filter_local, "value", None)
    if delivery_filter is None:
        delivery_filter = DeliveryControlFilter()
        _delivery_filter_local.value = delivery_filter
    visible_text = delivery_filter.feed(text)
    return _SPEECHABLE_WITH_CJK_PUNCTUATION.sub("", visible_text)


def split_streaming_sentences(text: str) -> list[str]:
    """Split completed Chinese sentences while retaining the unfinished tail.

    The upstream Responses handler uses NLTK's English Punkt tokenizer. It
    treats an entire Chinese reply containing ``。！？`` as one sentence, which
    makes its nominal LLM stream wait until completion before feeding TTS.
    A trailing empty item is intentional: the upstream loop then emits a
    sentence immediately when its closing punctuation arrives.
    """
    if re.search(r"[\u3400-\u9fff]", text):
        # Besides sentence endings, emit a short leading address/discourse
        # marker as soon as its Chinese comma arrives. Qwen otherwise receives
        # e.g. "三丰，小米……" as one long synthesis request and can flatten the
        # vocative pause into "三丰小米". Keeping the comma at the end of its own
        # early chunk lets the model's normal appended silence make the pause
        # audible without splitting every comma in the paragraph.
        boundaries: list[int] = []
        clause_start = 0
        for index, char in enumerate(text):
            if char in "。！？!?；：":
                boundaries.append(index + 1)
                clause_start = index + 1
            elif char == "…" and (index + 1 == len(text) or text[index + 1] != "…"):
                # A completed ellipsis is a punctuation boundary regardless of
                # the words around it; no semantic phrase list is involved.
                boundaries.append(index + 1)
                clause_start = index + 1
            elif char == "，":
                prefix = text[clause_start:index].strip()
                if 1 <= len(prefix) <= 8 and re.fullmatch(r"[\u3400-\u9fff]+", prefix):
                    boundaries.append(index + 1)
                    clause_start = index + 1
        if not boundaries:
            return [text]
        output: list[str] = []
        start = 0
        for end in sorted(set(boundaries)):
            if end <= start:
                continue
            output.append(text[start:end])
            start = end
        output.append(text[start:])
        return output
    from nltk import sent_tokenize as nltk_sent_tokenize

    return nltk_sent_tokenize(text)


def choose_tts_style(text: str, metadata: SenseVoiceMetadata | None = None) -> str:
    """Non-semantic fallback used only when the model omits its delivery plan."""

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
    if prosody_enabled:
        # Newlines are model-native whitespace, not SSML or invented spoken
        # punctuation. They strengthen existing semantic boundaries for the
        # Base voice-clone model while leaving transcript text untouched.
        # A short opening comma covers names such as "三丰，"; long clauses are
        # allowed to breathe at their own comma. ASCII commas remain untouched
        # so prices and identifiers such as 1,280.50 stay continuous.
        hinted: list[str] = []
        clause_chars = 0
        for index, char in enumerate(value):
            hinted.append(char)
            if char in "。！？!?；：":
                if index + 1 < len(value) and value[index + 1] != "\n":
                    hinted.append("\n")
                clause_chars = 0
                continue
            if char == "，":
                should_pause = clause_chars <= 8 or clause_chars >= max(12, max_clause_chars)
                if index + 1 < len(value) and not value[index + 1].isspace():
                    # Every Chinese comma receives at least a short whitespace
                    # cue. Vocatives and long clauses receive a stronger line
                    # boundary, while remaining a single synthesis request.
                    hinted.append("\n" if should_pause else " ")
                clause_chars = 0
                continue
            if char == "、":
                if index + 1 < len(value) and not value[index + 1].isspace():
                    hinted.append(" ")
                continue
            if not char.isspace():
                clause_chars += 1
        value = "".join(hinted)
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
        text, language_code, saw_end = super()._coalesce_pending_tts_input(current_input)
        metadata = get_turn_context(current_input.turn_id, current_input.turn_revision)
        cue = publish_expression(text, metadata=metadata)
        self._active_style = cue.style if self.emotion_control_enabled else "neutral"
        spoken_text = prepare_tts_text(
            text,
            self._active_style,
            prosody_enabled=self.prosody_enabled,
            max_clause_chars=self.prosody_max_clause_chars,
        )
        LOG.info(
            "TTS delivery source=%s style=%s expression=%s intensity=%.2f "
            "user_emotion=%s event=%s text=%r",
            cue.source,
            self._active_style,
            cue.profile,
            cue.intensity,
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
    handler_class = responses_module.ResponsesApiModelHandler
    if not getattr(handler_class, "_delivery_control_installed", False):
        original_apply_config = handler_class._apply_config

        def apply_config_with_delivery_control(self, chat, instructions):
            value = str(instructions or "")
            if "只逐字朗读用户提供的文字" not in value:
                value = f"{value}\n{DELIVERY_CONTROL_PROMPT}".strip()
            return original_apply_config(self, chat, value)

        handler_class._apply_config = apply_config_with_delivery_control
        handler_class._delivery_control_installed = True
    llm_utils_module.remove_unspeechable = remove_unspeechable_preserving_cjk
    # responses_api_language_model imports the helper into its module scope,
    # so patch that reference too. This fixes the actual punctuation loss
    # without guessing or rewriting any model output.
    responses_module.remove_unspeechable = remove_unspeechable_preserving_cjk
    responses_module.sent_tokenize = split_streaming_sentences
    LOG.info("Emotion-aware Qwen3-TTS and CJK punctuation preservation installed")
