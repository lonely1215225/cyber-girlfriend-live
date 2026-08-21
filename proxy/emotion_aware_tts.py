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
    "neutral": "像和熟人面对面聊天一样自然地说。按语义短句呼吸和停顿，有轻重音与语调变化，不要一口气念完，也不要像播报或朗诵。",
    "gentle": "用温柔、关心、真诚的语气说，语速稍慢。按语义短句轻轻停顿，重点柔和，像在认真安慰亲近的人，不要一口气念完。",
    "calm": "用平静、耐心、不对抗的语气说。声音柔和稳定，按意思自然停顿和强调，让对方感到被理解，不要机械播报。",
    "cheerful": "用明亮、亲切、带一点笑意的语气说。节奏轻快但有呼吸和停顿，重点词稍微上扬，不要夸张或一口气念完。",
    "serious": "用自然、清楚、稍微沉稳的聊天语气说。事实和重点之间自然停顿，不要像新闻播音，也不要把整段连成一句。",
}

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
_BREAK_WORDS = (
    "但是",
    "不过",
    "可是",
    "所以",
    "而且",
    "然后",
    "其实",
    "因为",
    "如果",
    "要不",
    "只是",
    "当然",
    "另外",
    "对了",
    "你可以",
    "我觉得",
)
_LEADING_FILLERS = (
    "说真的",
    "老实说",
    "对了",
    "哎呀",
    "哎",
    "诶",
    "欸",
    "嗯",
    "唔",
    "哦",
    "噢",
    "嘿",
    "哇",
    "哈哈",
    "好啦",
    "好呀",
    "对呀",
)
_QUESTION_CUES = (
    "为什么",
    "怎么",
    "什么",
    "哪个",
    "哪种",
    "哪里",
    "多少",
    "谁",
    "有没有",
    "是不是",
    "要不要",
    "想不想",
    "好不好",
    "行不行",
)
_PUNCTUATION = "，。！？!?；：……~～"
_EMPATHY_PIVOTS = ("别怕", "没事", "放心", "别急", "听我说", "我陪你")


def _clause_length(value: str, index: int) -> int:
    last = max((value.rfind(mark, 0, index) for mark in _PUNCTUATION), default=-1)
    return len(re.sub(r"\s+", "", value[last + 1 : index]))


def _insert_before(
    value: str, marker: str, *, minimum_before: int = 6, minimum_after: int = 4
) -> str:
    """Insert a comma before semantic pivots without splitting word fragments."""

    offset = 0
    while True:
        index = value.find(marker, offset)
        if index < 0:
            return value
        if (
            index > 0
            and value[index - 1] not in _PUNCTUATION
            and _clause_length(value, index) >= minimum_before
            and len(re.sub(r"\s+", "", value[index + len(marker) :])) >= minimum_after
            and not (marker == "所以" and value[index - 1] == "之")
        ):
            value = value[:index] + "，" + value[index:]
            offset = index + len(marker) + 1
        else:
            offset = index + len(marker)


def _restore_conversational_prosody(value: str, max_clause_chars: int) -> str:
    """Recover conservative spoken boundaries from weak punctuation-free LLM text."""

    for filler in _LEADING_FILLERS:
        if value.startswith(filler) and len(value) >= len(filler) + 3:
            tail = value[len(filler) :]
            if tail and tail[0] not in _PUNCTUATION:
                value = filler + "，" + tail
            break

    for marker in _BREAK_WORDS:
        value = _insert_before(value, marker)
    for marker in _EMPATHY_PIVOTS:
        value = _insert_before(value, marker, minimum_before=3, minimum_after=3)

    value = re.sub(
        r"你呢(?=(?:是不是|有没有|想不想|要不要|怎么|为什么))",
        "你呢，",
        value,
    )

    # A follow-up question often begins immediately after an unpunctuated
    # statement: “……魅力所在你最喜欢什么”. Give it its own intonation unit.
    question_start = re.compile(
        r"(?=(?:你|你们|大家)(?:呢，?)?(?:最|更|会|想|喜欢|觉得|有没有|是不是|要不要|想不想|怎么|为什么|在干嘛))"
    )
    offset = 1
    while match := question_start.search(value, offset):
        index = match.start()
        if _clause_length(value, index) >= 8 and value[index - 1] not in _PUNCTUATION:
            value = value[:index] + "。" + value[index:]
            offset = index + 2
        else:
            offset = index + 1

    # Long clauses get breathing points only at conversational particles or
    # safe phrase boundaries. Avoid blind fixed-width splitting, which can
    # separate a modifier from the word it describes.
    safe_endings = re.compile(
        r"(?:的话|的时候|一下|一会儿|没关系|不要紧|就好|就行|而已|[了呀吧嘛啦哦啊呢])"
    )
    parts = re.split(r"([，。！？!?；：])", value)
    rebuilt: list[str] = []
    for part in parts:
        if not part or part in _PUNCTUATION:
            rebuilt.append(part)
            continue
        segment = part
        cursor = 0
        while len(re.sub(r"\s+", "", segment[cursor:])) > max_clause_chars:
            candidates = [
                match.end()
                for match in safe_endings.finditer(segment, cursor + 6)
                if 8 <= match.end() - cursor <= max_clause_chars
                and len(segment) - match.end() >= 6
            ]
            if not candidates:
                break
            split_at = candidates[-1]
            segment = segment[:split_at] + "，" + segment[split_at:]
            cursor = split_at + 1
        rebuilt.append(segment)
    return "".join(rebuilt)


def _looks_like_question(value: str) -> bool:
    tail = re.split(r"[。！!；]", value)[-1]
    if any(cue in tail for cue in _QUESTION_CUES):
        return True
    return bool(re.search(r"(?:你|你们|大家).*(?:吗|么|嘛|呢)$", tail))


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
    """Improve spoken pauses without changing the browser-visible reply."""

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
        value = _restore_conversational_prosody(value, max(12, max_clause_chars))

    if value and value[-1] not in "。！？!?……~～":
        if _looks_like_question(value):
            value += "？"
        else:
            value += "！" if style == "cheerful" else "。"
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
        self.tts_temperature = float(os.environ.get("TTS_TEMPERATURE", "0.75"))
        self.tts_top_k = int(os.environ.get("TTS_TOP_K", "40"))
        self.tts_top_p = float(os.environ.get("TTS_TOP_P", "0.90"))
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

    def _process_voice_clone(self, text: str) -> Iterator[bytes | np.ndarray]:
        if self.backend != "faster_qwen3_tts":
            yield from super()._process_voice_clone(text)
            return

        max_new_tokens = self._estimate_max_new_tokens(text)
        instruct = None
        if self.style_instruct_enabled:
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
