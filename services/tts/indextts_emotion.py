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
# Keep energy low. High happy/angry makes IndexTTS-2.5 sound punched and loud.
_VECTORS: dict[str, list[float]] = {
    "happy": [0.32, 0.00, 0.00, 0.00, 0.00, 0.08, 0.00, 0.60],
    "playful": [0.18, 0.00, 0.00, 0.00, 0.00, 0.10, 0.06, 0.66],
    "warm": [0.06, 0.00, 0.00, 0.00, 0.00, 0.22, 0.00, 0.72],
    "tender": [0.04, 0.00, 0.04, 0.00, 0.00, 0.28, 0.00, 0.64],
    "shy": [0.03, 0.00, 0.00, 0.06, 0.00, 0.18, 0.00, 0.73],
    "serious": [0.00, 0.00, 0.00, 0.00, 0.00, 0.08, 0.00, 0.92],
    "neutral": [0.03, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.87],
    "sad": [0.00, 0.00, 0.55, 0.00, 0.00, 0.25, 0.00, 0.20],
    "angry": [0.00, 0.42, 0.00, 0.00, 0.08, 0.00, 0.00, 0.50],
    "surprised": [0.06, 0.00, 0.00, 0.08, 0.00, 0.00, 0.36, 0.50],
}

_VOICE_TEXT = {
    "happy": "轻轻笑着、软软地说",
    "playful": "小声、带点坏笑地说",
    "warm": "温柔、软软地说",
    "tender": "很轻、很温柔地说",
    "shy": "害羞、小声地说",
    "serious": "放轻声音、清楚地说",
    "neutral": "温柔、平静地说",
    "sad": "放轻、低落地说",
    "angry": "压着声音说，不要喊",
    "surprised": "轻轻惊讶，不要拔高",
}

_NONVERBAL_TEXT = {
    "soft_laugh": "带着很轻的笑软软地说",
    "laugh": "忍不住轻轻笑着、放软声音说",
    "sigh": "轻轻叹了口气说",
    "breath": "轻轻吸了口气再说",
    "hum": "轻轻哼着说",
}

_EMO_TEXT_INTENSITY = 0.45
_TENDER = "tender"


def clamp_duration_factor(value: Any, default: float = 1.02) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    # Slower is gentler. Do not let the director speed the clone up.
    return min(1.06, max(1.00, number))


def emo_alpha_from_intensity(intensity: Any) -> float:
    try:
        value = float(intensity)
    except (TypeError, ValueError):
        value = 0.0
    value = min(1.0, max(0.0, value))
    if value <= 0.18:
        return 0.08 + (0.16 - 0.08) * (value / 0.18 if value else 0.0)
    if value <= 0.45:
        return 0.16 + (0.26 - 0.16) * ((value - 0.18) / 0.27)
    if value <= 0.68:
        return 0.26 + (0.36 - 0.26) * ((value - 0.45) / 0.23)
    return 0.36 + (0.45 - 0.36) * min(1.0, (value - 0.68) / 0.32)


def emo_vector_for(vocal_emotion: str | None) -> list[float]:
    key = str(vocal_emotion or "neutral").strip().lower()
    return list(_VECTORS.get(key, _VECTORS["neutral"]))


def soften_vector(vocal_emotion: str | None, intensity: float) -> list[float]:
    """Pull everyday lines toward tender so happy/playful cannot punch the clone."""

    base = emo_vector_for(vocal_emotion)
    if intensity >= 0.45:
        return base
    tender = _VECTORS[_TENDER]
    mix = 0.62 if intensity <= 0.18 else 0.38
    return [round((1.0 - mix) * left + mix * right, 4) for left, right in zip(base, tender)]


def strip_english_bracket_tags(text: str) -> str:
    return FISH_INLINE_TAG_RE.sub("", str(text or "")).strip()


def plan_indextts_controls(
    *,
    vocal_emotion: str | None = None,
    vocal_intensity: float = 0.0,
    nonverbal: str | None = None,
    duration_factor: float = 1.02,
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
        "emo_vector": soften_vector(vocal_emotion, intensity),
        "emo_alpha": round(emo_alpha_from_intensity(intensity), 4),
        "use_emo_text": bool(use_text),
        "emo_text": emo_text,
        "duration_factor": clamp_duration_factor(duration_factor),
        "emo_audio": None,
    }
