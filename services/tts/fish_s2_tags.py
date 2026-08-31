"""Map delivery plans onto Fish S2 Pro inline tags and hide them from viewers."""

from __future__ import annotations

import re

# Fish accepts free-form English tags. Keep this conservative so ordinary
# Chinese brackets or citations are not stripped from the public room.
FISH_INLINE_TAG_RE = re.compile(r"\[[A-Za-z][^\[\]]{0,80}\]")

NONVERBAL_TAGS = {
    "soft_laugh": "[chuckle]",
    "laugh": "[laughing]",
    "sigh": "[sigh]",
    "breath": "[inhale]",
    "hum": "[humming]",
}

VOICE_TAGS = {
    "happy": "[laughing tone]",
    "playful": "[playful]",
    "warm": "[warm soft voice]",
    "tender": "[soft tender voice]",
    "shy": "[shy] [whisper]",
    "serious": "[serious]",
    "sad": "[sad]",
    "angry": "[angry]",
    "surprised": "[surprised]",
}


def strip_fish_inline_tags(text: str) -> str:
    """Remove complete Fish performance tags from public transcripts."""

    return FISH_INLINE_TAG_RE.sub("", str(text or ""))


def hide_incomplete_fish_tag(text: str) -> str:
    """Hide a trailing `[laugh` fragment until the closing bracket arrives."""

    value = str(text or "")
    opening = value.rfind("[")
    if opening < 0:
        return value
    tail = value[opening:]
    if "]" in tail:
        return value
    if re.match(r"\[[A-Za-z]", tail):
        return value[:opening]
    return value


def clean_public_fish_text(text: str) -> str:
    return hide_incomplete_fish_tag(strip_fish_inline_tags(text))


def _already_has_tag(text: str, tag: str) -> bool:
    existing = {match.group(0).lower().strip("[]") for match in FISH_INLINE_TAG_RE.finditer(text)}
    for part in re.findall(r"\[[^\]]+\]", tag):
        inner = part.lower().strip("[]")
        if any(inner == seen or inner in seen or seen in inner for seen in existing):
            return True
    return False


def apply_fish_performance_tags(
    text: str,
    *,
    vocal_emotion: str | None = None,
    vocal_intensity: float = 0.0,
    nonverbal: str | None = None,
) -> str:
    """Prefix Fish tags from the hidden delivery plan without duplicating LLM tags."""

    spoken = str(text or "").strip()
    if not spoken:
        return spoken
    prefixes: list[str] = []
    event = str(nonverbal or "none").strip().lower()
    event_tag = NONVERBAL_TAGS.get(event)
    if event_tag and not _already_has_tag(spoken, event_tag):
        prefixes.append(event_tag)
    emotion = str(vocal_emotion or "neutral").strip().lower()
    try:
        intensity = float(vocal_intensity or 0.0)
    except (TypeError, ValueError):
        intensity = 0.0
    if emotion != "neutral" and intensity >= 0.45:
        emotion_tag = VOICE_TAGS.get(emotion)
        if emotion_tag and not _already_has_tag(spoken, emotion_tag):
            prefixes.append(emotion_tag)
    if not prefixes:
        return spoken
    return f"{''.join(prefixes)}{spoken}"
