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

# This is a renderer protocol, not a semantic prompt template. The model
# chooses every value from the conversation as part of its existing response,
# avoiding a second inference pass before the first spoken sentence.
DELIVERY_CONTROL_PROMPT = """
你同时担任自己的实时表演导演。只在本轮要输出给观众的自然语言正文时，必须先输出一个隐藏控制标签：
<e profile intensity style>
profile 只能选 neutral、happy、surprised、serious、pout、one_brow、smirk、wink、cheek_puff、cute_annoyed、shy、laugh；intensity 是 0 到 1 的小数；style 只能选 neutral、gentle、calm、cheerful、serious。
你必须根据完整对话、真实语义、态度和表达节奏临场判断，不能按词语机械匹配。只有确实没有明显态度时才选 neutral；有明确情绪时应选择能自然呈现的相应表情，强度通常在 0.55 到 0.85。
每个回答的第一句前必须输出一个标签；后续句子的情绪、态度或表演节奏发生变化时，也在那句话前输出新标签。两句以上的回答通常安排二到四段自然变化，短回答不要为了变化而硬切。严格按三个值的顺序填写，不写属性名，不写结束标签；标签后立刻输出正文，不解释标签。调用工具或不输出正文时不要输出标签。
""".strip()


@dataclass(slots=True, frozen=True)
class DeliveryPlan:
    profile: str
    intensity: float
    style: str
    mouth_strength: float
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
    source: str
    delay_ms: int
    created_at: float


_lock = threading.Lock()
_sequence = 0
_plans: deque[DeliveryPlan] = deque(maxlen=32)
_cues: deque[ExpressionCue] = deque(maxlen=64)
_active_plan: DeliveryPlan | None = None
_response_epoch = 0
_timeline_base_offset: int | None = None
_PLAN_MAX_AGE_SECONDS = 20.0
_CONTROL_TAG = re.compile(
    r"^\s*([a-z_]+)\s+((?:0(?:\.\d+)?|1(?:\.0+)?))\s+"
    r"([a-z_]+)(?:\s+((?:0(?:\.\d+)?|1(?:\.0+)?)))?\s*$",
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
    r"(?=$|[\s，。！？、；：,.!?;:])",
    re.IGNORECASE,
)


def _number_prefix(value: str) -> bool:
    return bool(re.fullmatch(r"(?:[01](?:\.\d*)?)?", value))


def _bare_control_prefix(value: str) -> bool:
    """Whether an incomplete stream suffix can still become a control record."""

    candidate = value.strip()
    if not candidate or len(candidate) > 96:
        return False
    parts = candidate.split()
    profile = parts[0].lower()
    if len(parts) == 1:
        return any(item.startswith(profile) for item in EXPRESSION_PROFILES)
    if profile not in EXPRESSION_PROFILES or not _number_prefix(parts[1]):
        return False
    if len(parts) == 2:
        return True
    if not re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", parts[1]):
        return False
    style = parts[2].lower()
    if len(parts) == 3:
        return any(item.startswith(style) for item in DELIVERY_STYLES)
    if style not in DELIVERY_STYLES or len(parts) > 4:
        return False
    return _number_prefix(parts[3])


def submit_delivery_plan(
    profile: str,
    intensity: float,
    style: str,
    mouth_strength: float,
    *,
    text_offset: int = 0,
) -> bool:
    """Validate and enqueue a model-authored performance decision."""

    profile = str(profile or "").strip().lower()
    style = str(style or "").strip().lower()
    if profile not in EXPRESSION_PROFILES or style not in DELIVERY_STYLES:
        return False
    try:
        intensity_value = min(0.92, max(0.0, float(intensity)))
        mouth_value = min(0.40, max(0.0, float(mouth_strength)))
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
        for match in _BARE_CONTROL.finditer(combined):
            visible.append(combined[cursor:match.start()])
            submit_delivery_plan(
                match.group("profile"), float(match.group("intensity")),
                match.group("style"), float(match.group("mouth") or 0.0),
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
            search_start = max(0, len(remainder) - 96)
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
                match = _CONTROL_TAG.fullmatch(payload)
                if match:
                    submit_delivery_plan(
                        match.group(1), float(match.group(2)),
                        match.group(3), float(match.group(4) or 0.0),
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
                source="llm" if plan else "fallback",
                # Qwen3-TTS averages roughly 150-200ms per visible Chinese
                # character. The gateway binds this offset to its PCM clock,
                # so synthesis speed and network jitter cannot move the cue.
                delay_ms=(
                    min(12_000, max(0, plan.text_offset - base_offset) * 175)
                    if plan else 0
                ),
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
    "begin_delivery_generation", "begin_delivery_response", "clear_delivery_state", "cue_duration_ms", "cues_after",
    "publish_expression", "submit_delivery_plan",
]
