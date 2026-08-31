"""Emotion-aware Fish S2 Pro voice cloning with hidden delivery controls."""

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

from fish_s2_client import FishS2Client
from fish_s2_tags import apply_fish_performance_tags
from playback_policy import is_batch_tts
from voxcpm_shared_client import SharedVoxCPMClient
from expression_director import (
    DELIVERY_CONTROL_PROMPT,
    DeliveryControlFilter,
    begin_delivery_generation,
    publish_expression,
)
from output_harness import PublicOutputFilter

# Reuse the handler logger, which the service explicitly keeps at INFO. The
# module's former top-level logger stayed at WARNING, hiding the exact text and
# pause hints that reached TTS and making punctuation regressions hard to audit.
LOG = logging.getLogger("speech_to_speech.TTS.qwen3_tts_handler")

_SPEECHABLE_WITH_CJK_PUNCTUATION = re.compile(
    r"[^\w\s.,!?;:'\"\-()\/\\@#%&*+=$€£¥₹₽¢\[\]{}<>~`^|…—–，。！？；：、（）【】《》“”‘’￥]",
    flags=re.UNICODE,
)

_delivery_filter_local = threading.local()
_public_filter_local = threading.local()


def remove_unspeechable_preserving_cjk(text: str) -> str:
    """Hide LLM delivery controls and preserve the model's CJK punctuation."""

    delivery_filter = getattr(_delivery_filter_local, "value", None)
    if delivery_filter is None:
        delivery_filter = DeliveryControlFilter()
        _delivery_filter_local.value = delivery_filter
    visible_text = delivery_filter.feed(text)
    public_filter = getattr(_public_filter_local, "value", None)
    if public_filter is None:
        public_filter = PublicOutputFilter()
        _public_filter_local.value = public_filter
    visible_text = public_filter.feed(visible_text)
    return _SPEECHABLE_WITH_CJK_PUNCTUATION.sub("", visible_text)


_INTERJECTION_CORE = re.compile(
    r"^[嘿哎呦哟嗯喂哈啊哦额唔唉嗨哼诶欸哇]{1,4}[，。！？!?…\s]*$"
)
_CLOSING_QUOTE = frozenset("”」』\"'")
_MIN_QUESTION_FLUSH_CHARS = 8


def _spoken_core(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", str(text or "")).strip()


def _spoken_char_count(text: str) -> int:
    return len(re.sub(r"[\s。！？!?，、；：…—–“”‘’\"']+", "", _spoken_core(text)))


def is_leading_interjection(text: str) -> bool:
    """True for a lone 嘿/哎/呦 opener that must not become its own TTS clip."""

    core = _spoken_core(text)
    return bool(core) and bool(_INTERJECTION_CORE.fullmatch(core))


def _inside_chinese_quotes(text: str, index: int) -> bool:
    opens = text[: index + 1].count("“") + text[: index + 1].count("「")
    closes = text[: index + 1].count("”") + text[: index + 1].count("」")
    return opens > closes


def _is_dialogue_boundary(text: str, index: int, candidate: str) -> bool:
    """Keep quoted punchlines and tiny ？/！ clips attached to the next clause."""

    if is_leading_interjection(candidate):
        return False
    rest = text[index + 1 :].lstrip()
    if rest[:1] in _CLOSING_QUOTE or _inside_chinese_quotes(text, index):
        return False
    if text[index] in "？?!" and _spoken_char_count(candidate) < _MIN_QUESTION_FLUSH_CHARS:
        return False
    return True


def split_streaming_sentences(text: str) -> list[str]:
    """Split completed Chinese sentences while retaining the unfinished tail.

    The upstream Responses handler uses NLTK's English Punkt tokenizer. It
    treats an entire Chinese reply containing ``。！？`` as one sentence, which
    makes its nominal LLM stream wait until completion before feeding TTS.
    A trailing empty item is intentional: the upstream loop then emits a
    sentence immediately when its closing punctuation arrives.

    Dialogue flushes only on a real sentence end. Commas, ellipses, quoted
    questions, and tiny ？/！ openers stay inside the same request so a joke
    or "还有啊？" cannot play alone and then stall on the next clone. News /
    welcome connections set the batch policy so this returns the whole buffer.
    """
    if is_batch_tts():
        return [text]
    if re.search(r"[\u3400-\u9fff]", text):
        boundaries: list[int] = []
        start = 0
        for index, char in enumerate(text):
            if char not in "。！？!?":
                continue
            candidate = text[start : index + 1]
            if not _is_dialogue_boundary(text, index, candidate):
                continue
            boundaries.append(index + 1)
            start = index + 1
        if not boundaries:
            return [text]
        output: list[str] = []
        cursor = 0
        for end in boundaries:
            if end <= cursor:
                continue
            output.append(text[cursor:end])
            cursor = end
        output.append(text[cursor:])
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
    """Fish S2 handler that keeps the upstream Qwen3 TTS slot."""

    def setup(self, *args: Any, **kwargs: Any) -> None:
        self.emotion_control_enabled = os.environ.get("TTS_EMOTION_ENABLED", "1") != "0"
        self.style_instruct_enabled = (
            os.environ.get("TTS_STYLE_INSTRUCT_ENABLED", "1") != "0"
        )
        self.prosody_enabled = os.environ.get("TTS_PROSODY_ENABLED", "1") != "0"
        self.prosody_max_clause_chars = max(
            12, int(os.environ.get("TTS_PROSODY_MAX_CLAUSE_CHARS", "20"))
        )
        self.tts_temperature = float(os.environ.get("TTS_TEMPERATURE", "0.8"))
        self.tts_top_k = int(os.environ.get("TTS_TOP_K", "30"))
        self.tts_top_p = float(os.environ.get("TTS_TOP_P", "0.8"))
        self.tts_repetition_penalty = float(
            os.environ.get("TTS_REPETITION_PENALTY", "1.1")
        )
        self.tts_do_sample = os.environ.get("TTS_DO_SAMPLE", "1") != "0"
        self._active_style = "neutral"
        self.should_listen = kwargs.get("should_listen")
        self.cancel_scope = kwargs.get("cancel_scope")
        self.speculative_turns = kwargs.get("speculative_turns")
        self.ref_audio = kwargs.get("ref_audio") or os.environ.get("REF_AUDIO")
        self.ref_text = kwargs.get("ref_text") or os.environ.get("REF_TEXT", "")
        self.language = "chinese"
        self.speaker = kwargs.get("speaker")
        self.instruct = None
        self.xvec_only = False
        self.parity_mode = False
        self.non_streaming_mode = False
        self.max_new_tokens = int(kwargs.get("max_new_tokens") or 1024)
        self.blocksize = int(kwargs.get("blocksize") or 512)
        self.gen_kwargs = kwargs.get("gen_kwargs") or {}
        self.backend = os.environ.get("TTS_BACKEND", "fish_s2").strip() or "fish_s2"
        self.streaming_chunk_size = 4
        self.device = kwargs.get("device") or "cuda"
        self.model_name = os.environ.get("TTS_MODEL", "fish-s2-pro")
        self.model = None
        self._initial_speaker = self.speaker
        self._initial_ref_audio = self.ref_audio
        self.fish = None
        self.voxcpm = None
        if self.backend in {"voxcpm", "voxcpm_shared"}:
            self.voxcpm = SharedVoxCPMClient(
                base_url=os.environ.get("VOXCPM_SHARED_URL", "http://127.0.0.1:10102"),
                ref_audio=str(self.ref_audio or ""),
                ref_text=str(self.ref_text or ""),
            )
            self.voxcpm.wait_ready()
            LOG.info(
                "Emotion-aware shared VoxCPM ready url=%s ref=%s emotion=%s",
                self.voxcpm.base_url,
                self.ref_audio,
                self.emotion_control_enabled,
            )
        else:
            self.fish = FishS2Client(
                base_url=os.environ.get("FISH_S2_URL", "http://127.0.0.1:18781"),
                ref_audio=str(self.ref_audio or ""),
                ref_text=str(self.ref_text or ""),
            )
            self.fish.wait_ready()
            LOG.info(
                "Emotion-aware Fish S2 ready url=%s ref=%s emotion=%s",
                self.fish.base_url,
                self.ref_audio,
                self.emotion_control_enabled,
            )
        self.warmup()

    def warmup(self) -> None:
        if self.voxcpm is not None:
            # Do not steal their inference lock for a dummy sentence.
            LOG.info("Shared VoxCPM warmup skipped; using the already-loaded worker")
            return
        LOG.info("Warming up Fish S2 Pro")
        try:
            for _ in self._process_voice_clone("[laughing]嗯，我在。"):
                pass
            LOG.info("Fish S2 Pro warmed up")
        except Exception as exc:
            LOG.warning("Fish S2 warmup failed: %s", exc)

    def _model_type(self) -> str:
        return "base"

    def _coalesce_pending_tts_input(
        self, current_input: TTSInput
    ) -> tuple[str, str | None, bool]:
        batch = is_batch_tts()
        if batch:
            text, language_code, saw_end = super()._coalesce_pending_tts_input(
                current_input
            )
        else:
            # Dialogue synthesizes one finished sentence at a time. Do not
            # merge the next sentence into this request, and do not cut a
            # leading 嘿/哎 into its own generate.
            text = current_input.text
            language_code = current_input.language_code
            saw_end = False
        metadata = get_turn_context(current_input.turn_id, current_input.turn_revision)
        cue = publish_expression(text, metadata=metadata)
        self._active_style = cue.style if self.emotion_control_enabled else "neutral"
        spoken_text = prepare_tts_text(
            text,
            self._active_style,
            prosody_enabled=self.prosody_enabled,
            max_clause_chars=self.prosody_max_clause_chars,
        )
        if self.voxcpm is None:
            spoken_text = apply_fish_performance_tags(
                spoken_text,
                vocal_emotion=cue.vocal_emotion,
                vocal_intensity=cue.vocal_intensity,
                nonverbal=cue.nonverbal,
            )
        LOG.info(
            "TTS delivery source=%s style=%s expression=%s intensity=%.2f "
            "voice=%s voice_intensity=%.2f nonverbal=%s "
            "user_emotion=%s event=%s batch=%s text=%r",
            cue.source,
            self._active_style,
            cue.profile,
            cue.intensity,
            cue.vocal_emotion,
            cue.vocal_intensity,
            cue.nonverbal,
            metadata.emotion if metadata else None,
            metadata.event if metadata else None,
            batch,
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
        if self.voxcpm is not None:
            self.voxcpm.set_reference(str(self.ref_audio or ""), str(self.ref_text or ""))
            yield from self._stream(
                self.voxcpm.stream_clone(text),
                label=f"voxcpm_{self._active_style}",
            )
            return
        if self.fish is None:
            raise RuntimeError("no TTS backend is configured")
        self.fish.set_reference(str(self.ref_audio or ""), str(self.ref_text or ""))
        yield from self._stream(
            self.fish.stream_clone(text),
            label=f"fish_{self._active_style}",
        )


def install_emotion_aware_tts() -> None:
    import speech_to_speech.LLM.responses_api_language_model as responses_module
    import speech_to_speech.LLM.utils as llm_utils_module
    import speech_to_speech.TTS.qwen3_tts_handler as handler_module

    handler_module.Qwen3TTSHandler = EmotionAwareQwen3TTSHandler
    handler_class = responses_module.ResponsesApiModelHandler
    if not getattr(handler_class, "_delivery_control_installed", False):
        original_apply_config = handler_class._apply_config
        original_generate = handler_class._generate

        def apply_config_with_delivery_control(self, chat, instructions):
            value = str(instructions or "")
            if "只逐字朗读用户提供的文字" not in value:
                value = f"{value}\n{DELIVERY_CONTROL_PROMPT}".strip()
            return original_apply_config(self, chat, value)

        handler_class._apply_config = apply_config_with_delivery_control

        def generate_with_delivery_control(self, *args, **kwargs):
            # This runs in the same worker and immediately before the first
            # model delta, unlike the later outbound response.created event.
            begin_delivery_generation()
            _delivery_filter_local.value = DeliveryControlFilter()
            _public_filter_local.value = PublicOutputFilter()
            yield from original_generate(self, *args, **kwargs)

        handler_class._generate = generate_with_delivery_control
        handler_class._delivery_control_installed = True
    llm_utils_module.remove_unspeechable = remove_unspeechable_preserving_cjk
    # responses_api_language_model imports the helper into its module scope,
    # so patch that reference too. This fixes the actual punctuation loss
    # without guessing or rewriting any model output.
    responses_module.remove_unspeechable = remove_unspeechable_preserving_cjk
    responses_module.sent_tokenize = split_streaming_sentences
    LOG.info("Emotion-aware TTS and CJK punctuation preservation installed")
