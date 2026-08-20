"""Stable, emotion-aware Qwen3-TTS voice cloning with one reference audio."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from typing import Any

import numpy as np

from sensevoice_stt import SenseVoiceMetadata, get_turn_context
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler

LOG = logging.getLogger(__name__)

_STYLE_INSTRUCTIONS = {
    "gentle": "用温柔、关心、真诚的语气说，语速稍慢，停顿自然，像在认真安慰亲近的人。",
    "calm": "用平静、耐心、不对抗的语气说，声音柔和稳定，让对方感到被理解。",
    "cheerful": "用明亮、亲切、带一点笑意的语气说，轻快但不要夸张。",
    "serious": "用自然、清楚、稍微沉稳的语气说，重点明确，不要像机械播报。",
}

_GENTLE_WORDS = ("别怕", "没事", "我在", "抱抱", "辛苦", "难过", "委屈", "慢慢", "陪你", "心疼")
_CHEERFUL_WORDS = ("欢迎", "开心", "太好了", "恭喜", "哈哈", "嘿", "哇", "真棒", "好耶")
_SERIOUS_WORDS = ("新闻", "报道", "风险", "提醒", "目前", "数据显示", "需要注意", "局势", "发生")
_BREAK_WORDS = ("但是", "不过", "所以", "而且", "然后", "其实", "因为", "如果", "要不", "你可以", "我觉得")


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


def prepare_tts_text(text: str, style: str) -> str:
    """Improve spoken pauses without changing the browser-visible reply."""

    value = text.strip()
    value = re.sub(r"\.{3,}", "……", value)
    value = re.sub(r"…{3,}", "……", value)
    value = re.sub(r"[*_`#]+", "", value)
    value = re.sub(r"(?<=[\u3400-\u9fff，。！？；：……])\s+(?=[\u3400-\u9fff，。！？；：……])", "", value)
    value = re.sub(r"[ \t]+", " ", value).strip()

    # Very long punctuation-free Chinese clauses tend to become flat. Prefer
    # semantic conjunctions as breathing points; do not rewrite any words.
    if len(value) > 32 and not re.search(r"[，。！？；：]", value):
        for word in _BREAK_WORDS:
            index = value.find(word, 12)
            if 12 <= index <= len(value) - 8:
                value = value[:index] + "，" + value[index:]
                break

    if value and value[-1] not in "。！？!?……~～":
        value += "！" if style == "cheerful" else "。"
    return value


class EmotionAwareQwen3TTSHandler(Qwen3TTSHandler):
    """Qwen3 handler with bounded sampling and per-turn acoustic guidance."""

    def setup(self, *args: Any, **kwargs: Any) -> None:
        self.emotion_control_enabled = os.environ.get("TTS_EMOTION_ENABLED", "1") != "0"
        self.style_instruct_enabled = os.environ.get("TTS_STYLE_INSTRUCT_ENABLED", "1") != "0"
        self.tts_temperature = float(os.environ.get("TTS_TEMPERATURE", "0.75"))
        self.tts_top_k = int(os.environ.get("TTS_TOP_K", "40"))
        self.tts_top_p = float(os.environ.get("TTS_TOP_P", "0.90"))
        self.tts_repetition_penalty = float(os.environ.get("TTS_REPETITION_PENALTY", "1.05"))
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

    def _coalesce_pending_tts_input(self, current_input: TTSInput) -> tuple[str, str | None, bool]:
        text, language_code, saw_end = super()._coalesce_pending_tts_input(current_input)
        metadata = get_turn_context(current_input.turn_id, current_input.turn_revision)
        self._active_style = choose_tts_style(text, metadata) if self.emotion_control_enabled else "neutral"
        spoken_text = prepare_tts_text(text, self._active_style)
        LOG.info(
            "TTS delivery style=%s user_emotion=%s event=%s text=%s",
            self._active_style,
            metadata.emotion if metadata else None,
            metadata.event if metadata else None,
            spoken_text,
        )
        return spoken_text, language_code, saw_end

    def _process_voice_clone(self, text: str) -> Iterator[bytes | np.ndarray]:
        if self.backend != "faster_qwen3_tts":
            yield from super()._process_voice_clone(text)
            return

        max_new_tokens = self._estimate_max_new_tokens(text)
        instruct = None
        if self.style_instruct_enabled and self._active_style != "neutral":
            instruct = _STYLE_INSTRUCTIONS.get(self._active_style)

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
    import speech_to_speech.TTS.qwen3_tts_handler as handler_module

    handler_module.Qwen3TTSHandler = EmotionAwareQwen3TTSHandler
    LOG.info("Emotion-aware Qwen3-TTS adapter installed")
