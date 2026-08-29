"""LLM-authored delivery controls shared by the realtime LLM, TTS and AVTR.

The language model emits one compact, hidden control record before its spoken
answer. This module validates that renderer protocol and queues it for TTS. It
deliberately contains no phrase or keyword-to-expression rules: semantic
judgement belongs to the model.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass


EXPRESSION_PROFILES = frozenset({
    "neutral", "happy", "surprised", "serious", "pout", "one_brow",
    "smirk", "wink", "cheek_puff", "cute_annoyed", "shy", "laugh",
})
DELIVERY_STYLES = frozenset({"neutral", "gentle", "calm", "cheerful", "serious"})
VOCAL_EMOTIONS = frozenset({
    "neutral", "happy", "playful", "warm", "tender", "shy", "serious",
    "sad", "angry", "surprised",
})
NONVERBAL_EVENTS = frozenset({
    "none", "soft_laugh", "laugh", "sigh", "breath", "hum",
})

# This is a renderer protocol, not a semantic prompt template. The model
# chooses every value from the conversation as part of its existing response,
# avoiding a second inference pass before the first spoken sentence.
DELIVERY_CONTROL_PROMPT = """
你同时担任自己的实时表演导演。只在本轮要输出给观众的自然语言正文时，必须先输出一个隐藏控制标签：
<e face face_intensity voice voice_intensity nonverbal pace>
face 只能选 neutral、happy、surprised、serious、pout、one_brow、smirk、wink、cheek_puff、cute_annoyed、shy、laugh；face_intensity 是 0 到 1。
voice 只能选 neutral、happy、playful、warm、tender、shy、serious、sad、angry、surprised；voice_intensity 是 0 到 1；nonverbal 只能选 none、soft_laugh、laugh、sigh、breath、hum；pace 是 0.96 到 1.04。
面部表情和声音情绪必须分别判断。根据完整对话、真实语义、态度和表达节奏临场选择，不能按词语机械匹配。面部可以灵动，但声音平时必须保持接近角色原始参考音色：普通说明、事实陈述、新闻主体和日常衔接句优先用 neutral，voice_intensity 填 0.00 到 0.18，pace 填 0.98 到 1.02。不要为了显得活泼而让整段持续高音、高亢或撒娇。
只有某一个具体句子确实承载明显的惊喜、调侃、害羞、安慰、生气、难过或强调时，才为该情绪片段选择对应 voice，并将 voice_intensity 提高到 0.45 到 0.68；极少数情绪高潮才可超过 0.68。特殊片段结束后，必须在下一普通句前重新输出 neutral 的低强度标签，不能让上一句的高情绪沿用到整段回答。面部表情不要求随声音一起升高。
只有真的需要可听笑声、叹气或呼吸时才选 nonverbal，不要每句都加。严肃新闻、灾难、求助和难过话题不强行撒娇或发笑。
选择 soft_laugh、laugh、sigh 或 hum 时，正文必须自然写出当下真会说出口的简短语气或拟声，让听众真正听得到。例如轻笑可以是“嘿嘿，被你发现了。”，叹气可以是“唉……那我慢慢说。”；只学表达方式，不照抄。不得写括号舞台提示、英文动作标签或控制词。
每个回答的第一句前必须输出一个标签；后续只有特殊情绪片段开始或结束时才输出新标签。把连续的普通句视为同一低情绪片段，特殊句结束后务必切回低情绪。不要把“诶、哟、嘿嘿”等单独作为一个待合成句子，应自然连到后面的正文。严格按六个值的顺序填写，不写属性名，不写结束标签。例如普通句用 <e happy 0.58 neutral 0.08 none 1.00>，只有确实需要调侃的句子才用 <e smirk 0.62 playful 0.56 none 1.00>。标签后立刻输出正文，不解释标签。调用工具或不输出正文时不要输出标签。
正文只能是可直接展示和朗读的纯文本，禁止Markdown和任何HTML/XML标签，包括<br>、<p>、<div>。
""".strip()


@dataclass(slots=True, frozen=True)
class DeliveryPlan:
    profile: str
    intensity: float
    style: str
    mouth_strength: float
    vocal_emotion: str
    vocal_intensity: float
    nonverbal: str
    duration_factor: float
    text_offset: int
    created_at: float


@dataclass(slots=True, frozen=True)
class ExpressionCue:
    sequence: int
    profile: str
    intensity: float
    duration_ms: int
    mouth_strength: float
    style: str
    vocal_emotion: str
    vocal_intensity: float
    nonverbal: str
    duration_factor: float
    source: str
    delay_ms: int
    response_epoch: int
    created_at: float


_lock = threading.Lock()
_sequence = 0
_plans: deque[DeliveryPlan] = deque(maxlen=32)
_cues: deque[ExpressionCue] = deque(maxlen=64)
_active_plan: DeliveryPlan | None = None
_response_epoch = 0
_timeline_base_offset: int | None = None
_PLAN_MAX_AGE_SECONDS = 20.0
_CONTROL_COMPACT = re.compile(
    r"^\s*([a-z_]+)\s+((?:0(?:\.\d+)?|1(?:\.0+)?))\s+([a-z_]+)\s+"
    r"((?:0(?:\.\d+)?|1(?:\.0+)?))\s+([a-z_]+)\s+"
    r"((?:0(?:\.\d+)?|1(?:\.\d+)?))\s*$",
    re.IGNORECASE,
)
_CONTROL_TAG = re.compile(
    r"^\s*([a-z_]+)\s+((?:0(?:\.\d+)?|1(?:\.0+)?))\s+"
    r"([a-z_]+)(?:\s+((?:0(?:\.\d+)?|1(?:\.0+)?)))?"
    r"(?:\s+([a-z_]+)\s+((?:0(?:\.\d+)?|1(?:\.0+)?))\s+([a-z_]+)\s+"
    r"((?:0(?:\.\d+)?|1(?:\.\d+)?)))?\s*$",
    re.IGNORECASE,
)
_CONTROL_ATTRIBUTE = re.compile(
    r"([a-z_]+)\s*=\s*[\"']?([a-z_]+|(?:0(?:\.\d+)?|1(?:\.0+)?))[\"']?",
    re.IGNORECASE,
)
_BARE_CONTROL = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<profile>{'|'.join(sorted(EXPRESSION_PROFILES, key=len, reverse=True))})"
    rf"\s+(?P<intensity>(?:0(?:\.\d+)?|1(?:\.0+)?))"
    rf"\s+(?P<style>{'|'.join(sorted(DELIVERY_STYLES, key=len, reverse=True))})"
    rf"(?:\s+(?P<mouth>(?:0(?:\.\d+)?|1(?:\.0+)?)))?"
    rf"(?:\s+(?P<vocal>{'|'.join(sorted(VOCAL_EMOTIONS, key=len, reverse=True))})"
    rf"\s+(?P<vocal_intensity>(?:0(?:\.\d+)?|1(?:\.0+)?))"
    rf"\s+(?P<nonverbal>{'|'.join(sorted(NONVERBAL_EVENTS, key=len, reverse=True))})"
    rf"\s+(?P<pace>(?:0(?:\.\d+)?|1(?:\.\d+)?)))?"
    r"(?!\.\d)(?=$|[\r\n，。！？、；：,.!?;:])",
    re.IGNORECASE,
)
_BARE_COMPACT_CONTROL = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<profile>{'|'.join(sorted(EXPRESSION_PROFILES, key=len, reverse=True))})"
    rf"\s+(?P<intensity>(?:0(?:\.\d+)?|1(?:\.0+)?))"
    rf"\s+(?P<vocal>{'|'.join(sorted(VOCAL_EMOTIONS, key=len, reverse=True))})"
    rf"\s+(?P<vocal_intensity>(?:0(?:\.\d+)?|1(?:\.0+)?))"
    rf"\s+(?P<nonverbal>{'|'.join(sorted(NONVERBAL_EVENTS, key=len, reverse=True))})"
    rf"\s+(?P<pace>(?:0(?:\.\d+)?|1(?:\.\d+)?))"
    r"(?!\.\d)(?=$|[\r\n，。！？、；：,.!?;:])",
    re.IGNORECASE,
)


def _number_prefix(value: str) -> bool:
    return bool(re.fullmatch(r"(?:[01](?:\.\d*)?)?", value))


def _style_for_vocal(vocal: str) -> str:
    return {
        "happy": "cheerful", "playful": "cheerful", "warm": "gentle",
        "tender": "gentle", "shy": "gentle", "serious": "serious",
        "sad": "calm", "angry": "serious", "surprised": "cheerful",
    }.get(vocal, "neutral")


def _matches_prefix(parts: list[str], enums: dict[int, frozenset[str]], length: int) -> bool:
    if len(parts) > length:
        return False
    for index, part in enumerate(parts):
        lowered = part.lower()
        if index in enums:
            choices = enums[index]
            if index == len(parts) - 1:
                if not any(item.startswith(lowered) for item in choices):
                    return False
            elif lowered not in choices:
                return False
        elif not _number_prefix(part):
            return False
    return True


def _bare_control_prefix(value: str) -> bool:
    """Whether an incomplete stream suffix can still become a control record."""

    candidate = value.strip()
    if not candidate or len(candidate) > 180:
        return False
    parts = candidate.split()
    compact = {0: EXPRESSION_PROFILES, 2: VOCAL_EMOTIONS, 4: NONVERBAL_EVENTS}
    extended = {0: EXPRESSION_PROFILES, 2: DELIVERY_STYLES, 4: VOCAL_EMOTIONS, 6: NONVERBAL_EVENTS}
    legacy = {0: EXPRESSION_PROFILES, 2: DELIVERY_STYLES}
    return (
        _matches_prefix(parts, compact, 6)
        or _matches_prefix(parts, extended, 8)
        or _matches_prefix(parts, legacy, 4)
    )


def submit_delivery_plan(
    profile: str,
    intensity: float,
    style: str,
    mouth_strength: float,
    vocal_emotion: str | None = None,
    vocal_intensity: float | None = None,
    nonverbal: str = "none",
    duration_factor: float = 1.0,
    *,
    text_offset: int = 0,
) -> bool:
    """Validate and enqueue a model-authored performance decision."""

    profile = str(profile or "").strip().lower()
    style = str(style or "").strip().lower()
    vocal_emotion = str(vocal_emotion or style or "neutral").strip().lower()
    vocal_emotion = {
        "gentle": "tender", "calm": "neutral", "cheerful": "happy",
    }.get(vocal_emotion, vocal_emotion)
    nonverbal = str(nonverbal or "none").strip().lower()
    if profile not in EXPRESSION_PROFILES or style not in DELIVERY_STYLES:
        return False
    if vocal_emotion not in VOCAL_EMOTIONS or nonverbal not in NONVERBAL_EVENTS:
        return False
    try:
        intensity_value = min(0.92, max(0.0, float(intensity)))
        mouth_value = min(0.40, max(0.0, float(mouth_strength)))
        vocal_value = min(0.92, max(0.0, float(
            intensity_value if vocal_intensity is None else vocal_intensity
        )))
        # Large duration changes alter perceived pitch and speaker identity.
        # Semantic pacing still comes from the LLM, but the renderer keeps it
        # inside a voice-safe range around the reference recording.
        duration_value = min(1.04, max(0.96, float(duration_factor)))
    except (TypeError, ValueError):
        return False
    if profile == "neutral":
        intensity_value = 0.0
        mouth_value = 0.0
    with _lock:
        _plans.append(DeliveryPlan(
            profile=profile,
            intensity=intensity_value,
            style=style,
            mouth_strength=mouth_value,
            vocal_emotion=vocal_emotion,
            vocal_intensity=vocal_value,
            nonverbal=nonverbal,
            duration_factor=duration_value,
            text_offset=max(0, int(text_offset)),
            created_at=time.monotonic(),
        ))
    return True


class DeliveryControlFilter:
    """Strip split hidden control tags and enqueue valid LLM decisions."""

    def __init__(self) -> None:
        self.buffer = ""
        self.plain_pending = ""
        self.visible_chars = 0
        self.response_epoch = _response_epoch

    def _sync_response_epoch(self) -> None:
        if self.response_epoch == _response_epoch:
            return
        self.buffer = ""
        self.plain_pending = ""
        self.visible_chars = 0
        self.response_epoch = _response_epoch

    def _filter_plain(self, piece: str, *, final: bool, output_offset: int) -> str:
        """Separate bare renderer records from arbitrary streamed prose.

        Models occasionally omit both angle brackets and line boundaries, so
        controls must be recognized at every text position. Only the exact
        renderer grammar is consumed; semantic expression selection remains
        entirely model-authored.
        """

        combined = self.plain_pending + str(piece or "")
        self.plain_pending = ""
        visible: list[str] = []
        cursor = 0
        matches = [(match, True) for match in _BARE_COMPACT_CONTROL.finditer(combined)]
        matches.extend((match, False) for match in _BARE_CONTROL.finditer(combined))
        matches.sort(key=lambda item: (item[0].start(), -(item[0].end() - item[0].start())))
        for match, compact in matches:
            if match.start() < cursor:
                continue
            # A 3/4-field legacy match can steal the first tokens of a 6-field
            # compact record while it is still streaming, especially when the
            # third token is also a delivery style such as "neutral".
            if not compact and not final:
                after = combined[match.end():]
                if not after.strip() or re.match(r"[ \t]+[A-Za-z0-9_.]", after):
                    visible.append(combined[cursor:match.start()])
                    self.plain_pending = combined[match.start():]
                    return "".join(visible)
            visible.append(combined[cursor:match.start()])
            if compact:
                vocal = match.group("vocal")
                submit_delivery_plan(
                    match.group("profile"), float(match.group("intensity")),
                    _style_for_vocal(vocal), 0.0, vocal,
                    float(match.group("vocal_intensity")),
                    match.group("nonverbal"), float(match.group("pace")),
                    text_offset=self.visible_chars + output_offset + sum(len(v) for v in visible),
                )
            else:
                submit_delivery_plan(
                    match.group("profile"), float(match.group("intensity")),
                    match.group("style"), float(match.group("mouth") or 0.0),
                    match.group("vocal"),
                    float(match.group("vocal_intensity")) if match.group("vocal_intensity") else None,
                    match.group("nonverbal") or "none",
                    float(match.group("pace") or 1.0),
                    text_offset=self.visible_chars + output_offset + sum(len(v) for v in visible),
                )
            cursor = match.end()
            line_start = combined.rfind("\n", 0, match.start()) + 1
            if not combined[line_start:match.start()].strip():
                newline = re.match(r"[ \t]*(?:\r?\n)+", combined[cursor:])
                if newline:
                    cursor += newline.end()
        remainder = combined[cursor:]
        if not final and remainder:
            search_start = max(0, len(remainder) - 180)
            for token in re.finditer(
                r"(?<![A-Za-z0-9_])[A-Za-z_]", remainder[search_start:]
            ):
                index = search_start + token.start()
                if _bare_control_prefix(remainder[index:]):
                    visible.append(remainder[:index])
                    self.plain_pending = remainder[index:]
                    remainder = ""
                    break
        visible.append(remainder)
        return "".join(visible)

    def feed(self, piece: str, *, final: bool = False) -> str:
        self._sync_response_epoch()
        self.buffer += str(piece or "")
        output: list[str] = []
        while self.buffer:
            opening = self.buffer.find("<")
            if opening < 0:
                output.append(self._filter_plain(
                    self.buffer, final=final,
                    output_offset=sum(len(part) for part in output),
                ))
                self.buffer = ""
                break
            if opening > 0:
                output.append(self._filter_plain(
                    self.buffer[:opening], final=False,
                    output_offset=sum(len(part) for part in output),
                ))
                self.buffer = self.buffer[opening:]
            closing = self.buffer.find(">")
            if closing < 0:
                prefix = self.buffer.lower()
                plausible_control = prefix == "<" or prefix.startswith("<e")
                if final or not plausible_control:
                    output.append(self.buffer)
                    self.buffer = ""
                break
            raw = self.buffer[: closing + 1]
            body = self.buffer[1:closing]
            stripped_body = body.lstrip()
            lowered_body = stripped_body.lower()
            if lowered_body == "/e":
                pass
            elif lowered_body.startswith("e "):
                payload = stripped_body[2:]
                compact_match = _CONTROL_COMPACT.fullmatch(payload)
                match = _CONTROL_TAG.fullmatch(payload)
                if compact_match:
                    vocal = compact_match.group(3).lower()
                    submit_delivery_plan(
                        compact_match.group(1), float(compact_match.group(2)),
                        _style_for_vocal(vocal), 0.0, vocal,
                        float(compact_match.group(4)), compact_match.group(5),
                        float(compact_match.group(6)),
                        text_offset=self.visible_chars + sum(len(part) for part in output),
                    )
                elif match:
                    submit_delivery_plan(
                        match.group(1), float(match.group(2)),
                        match.group(3), float(match.group(4) or 0.0),
                        match.group(5),
                        float(match.group(6)) if match.group(6) else None,
                        match.group(7) or "none",
                        float(match.group(8) or 1.0),
                        text_offset=self.visible_chars + sum(len(part) for part in output),
                    )
                else:
                    # Small local models sometimes serialize the requested
                    # structure as XML attributes. Accept that serialization
                    # without deriving any semantic value from the dialogue.
                    attrs = {key.lower(): value.lower() for key, value in _CONTROL_ATTRIBUTE.findall(payload)}
                    if {"profile", "intensity", "style"}.issubset(attrs):
                        submit_delivery_plan(
                            attrs["profile"], float(attrs["intensity"]),
                            attrs["style"], float(attrs.get("mouth", 0.0)),
                            attrs.get("voice") or attrs.get("vocal"),
                            float(attrs["voice_intensity"]) if attrs.get("voice_intensity") else None,
                            attrs.get("nonverbal", "none"),
                            float(attrs.get("pace", 1.0)),
                            text_offset=self.visible_chars + sum(len(part) for part in output),
                        )
                # Malformed control records are hidden too; they must never be
                # displayed in chat or pronounced by TTS.
            else:
                output.append(raw)
            self.buffer = self.buffer[closing + 1:]
        if final and self.plain_pending:
            output.append(self._filter_plain(
                "", final=True, output_offset=sum(len(part) for part in output)
            ))
        visible = "".join(output)
        self.visible_chars += len(visible)
        return visible


def _take_plans() -> list[DeliveryPlan]:
    global _active_plan
    now = time.monotonic()
    with _lock:
        while _plans and now - _plans[0].created_at > _PLAN_MAX_AGE_SECONDS:
            _plans.popleft()
        pending = list(_plans)
        _plans.clear()
        if pending:
            _active_plan = pending[-1]
            return pending
        return [_active_plan] if _active_plan is not None else []


def cue_duration_ms(text: str, profile: str) -> int:
    """Estimate a safe visual hold from spoken length, without semantic rules."""

    compact_length = len(re.sub(r"\s+", "", str(text or "")))
    estimate = max(900, min(5200, 500 + compact_length * 150))
    if profile == "wink":
        # A blink-sized cue disappeared inside the source-image crossfade.
        # Keep a model-directed wink visible long enough to ease in and out.
        return max(1600, min(2400, estimate))
    return estimate


def publish_expression(text: str, *_args, **_kwargs) -> ExpressionCue:
    """Bind the next LLM plan to TTS, or use a non-semantic neutral fallback."""

    global _sequence, _timeline_base_offset
    plans = _take_plans()
    with _lock:
        if not plans:
            plans = [None]
        if _timeline_base_offset is None and plans[0] is not None:
            _timeline_base_offset = plans[0].text_offset
        base_offset = _timeline_base_offset or 0
        emitted: list[ExpressionCue] = []
        for plan in plans:
            _sequence += 1
            profile = plan.profile if plan else "neutral"
            cue = ExpressionCue(
                sequence=_sequence,
                profile=profile,
                intensity=plan.intensity if plan else 0.0,
                duration_ms=cue_duration_ms(text, profile),
                mouth_strength=plan.mouth_strength if plan else 0.0,
                style=plan.style if plan else "neutral",
                vocal_emotion=plan.vocal_emotion if plan else "neutral",
                vocal_intensity=plan.vocal_intensity if plan else 0.0,
                nonverbal=plan.nonverbal if plan else "none",
                duration_factor=plan.duration_factor if plan else 1.0,
                source="llm" if plan else "fallback",
                # Qwen3-TTS averages roughly 150-200ms per visible Chinese
                # character. The gateway binds this offset to its PCM clock,
                # so synthesis speed and network jitter cannot move the cue.
                delay_ms=(
                    min(12_000, max(0, plan.text_offset - base_offset) * 175)
                    if plan else 0
                ),
                response_epoch=_response_epoch,
                created_at=time.monotonic(),
            )
            _cues.append(cue)
            emitted.append(cue)
        return emitted[0]


def cues_after(sequence: int) -> list[ExpressionCue]:
    with _lock:
        return [cue for cue in _cues if cue.sequence > sequence]


def clear_delivery_state() -> None:
    """Discard unconsumed controls after a session interruption."""

    global _active_plan, _timeline_base_offset
    with _lock:
        _plans.clear()
        _active_plan = None
        _timeline_base_offset = None


def begin_delivery_response() -> None:
    """Mark outbound response delivery without discarding parsed controls.

    The realtime server can publish ``response.created`` after its generation
    worker has already parsed the first tokens. Resetting here used to erase a
    valid cue and exposed a bracket-less remainder to speech.
    """

    global _active_plan
    with _lock:
        if not _plans:
            _active_plan = None


def begin_delivery_generation() -> None:
    """Reset protocol state at the true start of an LLM generation worker."""

    global _active_plan, _response_epoch, _timeline_base_offset
    with _lock:
        _response_epoch += 1
        _plans.clear()
        _active_plan = None
        _timeline_base_offset = None


__all__ = [
    "DELIVERY_CONTROL_PROMPT", "DeliveryControlFilter", "ExpressionCue",
    "VOCAL_EMOTIONS", "NONVERBAL_EVENTS",
    "begin_delivery_generation", "begin_delivery_response", "clear_delivery_state", "cue_duration_ms", "cues_after",
    "publish_expression", "submit_delivery_plan",
]
