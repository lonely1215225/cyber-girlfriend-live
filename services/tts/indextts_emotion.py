"""Map hidden delivery cues onto IndexTTS-2.5 emotion controls."""

from __future__ import annotations

from typing import Any

from fish_s2_tags import FISH_INLINE_TAG_RE

INDEX_EMOTION_ORDER = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)

# Index order: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
_VECTORS: dict[str, list[float]] = {
    "happy": [0.85, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.15],
    "playful": [0.55, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.35],
    "warm": [0.15, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.65],
    "tender": [0.10, 0.00, 0.00, 0.00, 0.00, 0.25, 0.00, 0.65],
    "shy": [0.05, 0.00, 0.00, 0.08, 0.00, 0.15, 0.00, 0.72],
    "serious": [0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.95],
    "neutral": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    "sad": [0.00, 0.00, 0.80, 0.00, 0.00, 0.15, 0.00, 0.05],
    "angry": [0.00, 0.80, 0.00, 0.00, 0.10, 0.00, 0.00, 0.10],
    "surprised": [0.10, 0.00, 0.00, 0.10, 0.00, 0.00, 0.75, 0.05],
}

_VOICE_TEXT = {
    "happy": "开心、轻快地说",
    "playful": "俏皮、带点坏笑地说",
    "warm": "温暖、柔和地说",
    "tender": "温柔、安慰地说",
    "shy": "害羞、小声地说",
    "serious": "认真、清楚地说明",
    "neutral": "自然、平静地说",
    "sad": "难过、低落地说",
    "angry": "生气、但不要破音",
    "surprised": "惊讶、短促地反应",
}

_NONVERBAL_TEXT = {
    "soft_laugh": "带着轻笑说",
    "laugh": "忍不住轻轻笑着说",
    "sigh": "叹了口气说",
    "breath": "吸了口气再说",
    "hum": "轻轻哼着说",
}

_EMO_TEXT_INTENSITY = 0.35


def clamp_duration_factor(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(1.04, max(0.96, number))


def emo_alpha_from_intensity(intensity: Any) -> float:
    try:
        value = float(intensity)
    except (TypeError, ValueError):
        value = 0.0
    value = min(1.0, max(0.0, value))
    if value <= 0.18:
        return 0.15 + (0.28 - 0.15) * (value / 0.18 if value else 0.0)
    if value <= 0.45:
        return 0.28 + (0.45 - 0.28) * ((value - 0.18) / 0.27)
    if value <= 0.68:
        return 0.45 + (0.65 - 0.45) * ((value - 0.45) / 0.23)
    return 0.70 + (0.75 - 0.70) * min(1.0, (value - 0.68) / 0.32)


def emo_vector_for(vocal_emotion: str | None) -> list[float]:
    key = str(vocal_emotion or "neutral").strip().lower()
    return list(_VECTORS.get(key, _VECTORS["neutral"]))


def strip_english_bracket_tags(text: str) -> str:
    return FISH_INLINE_TAG_RE.sub("", str(text or "")).strip()


def plan_indextts_controls(
    *,
    vocal_emotion: str | None = None,
    vocal_intensity: float = 0.0,
    nonverbal: str | None = None,
    duration_factor: float = 1.0,
    spoken_text: str = "",
) -> dict[str, Any]:
    """Build Index infer kwargs. Official text control overwrites the vector."""

    try:
        intensity = float(vocal_intensity)
    except (TypeError, ValueError):
        intensity = 0.0
    event = str(nonverbal or "none").strip().lower()
    spoken = strip_english_bracket_tags(spoken_text)
    use_text = intensity >= _EMO_TEXT_INTENSITY or event not in {"", "none"}
    emo_text = ""
    if use_text:
        emo_text = _NONVERBAL_TEXT.get(event) or _VOICE_TEXT.get(
            str(vocal_emotion or "neutral").strip().lower(),
            _VOICE_TEXT["neutral"],
        )
    return {
        "spoken_text": spoken,
        "emo_vector": emo_vector_for(vocal_emotion),
        "emo_alpha": round(emo_alpha_from_intensity(intensity), 4),
        "use_emo_text": bool(use_text),
        "emo_text": emo_text,
        "duration_factor": clamp_duration_factor(duration_factor),
        "emo_audio": None,
    }
