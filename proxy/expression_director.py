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
你必须根据完整对话、真实语义、态度和表达节奏临场判断，不能按词语机械匹配。只有确实没有明显态度时才选 neutral；有明确情绪时应选择能自然呈现的相应表情，强度通常在 0.55 到 0.85。每次回答只在最开头输出一个标签，严格按三个值的顺序填写，不写属性名，不写结束标签；标签后立刻输出正文，不解释标签。调用工具或不输出正文时不要输出标签。
""".strip()


@dataclass(slots=True, frozen=True)
class DeliveryPlan:
    profile: str
    intensity: float
    style: str
    mouth_strength: float
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
    created_at: float


_lock = threading.Lock()
_sequence = 0
_plans: deque[DeliveryPlan] = deque(maxlen=32)
_cues: deque[ExpressionCue] = deque(maxlen=64)
_active_plan: DeliveryPlan | None = None
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


def submit_delivery_plan(profile: str, intensity: float, style: str, mouth_strength: float) -> bool:
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
            created_at=time.monotonic(),
        ))
    return True


class DeliveryControlFilter:
    """Strip split hidden control tags and enqueue valid LLM decisions."""

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, piece: str, *, final: bool = False) -> str:
        self.buffer += str(piece or "")
        output: list[str] = []
        while self.buffer:
            opening = self.buffer.find("<")
            if opening < 0:
                output.append(self.buffer)
                self.buffer = ""
                break
            if opening > 0:
                output.append(self.buffer[:opening])
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
                        )
                # Malformed control records are hidden too; they must never be
                # displayed in chat or pronounced by TTS.
            else:
                output.append(raw)
            self.buffer = self.buffer[closing + 1:]
        return "".join(output)


def _take_plan() -> DeliveryPlan | None:
    global _active_plan
    now = time.monotonic()
    with _lock:
        if _active_plan is not None:
            return _active_plan
        while _plans and now - _plans[0].created_at > _PLAN_MAX_AGE_SECONDS:
            _plans.popleft()
        _active_plan = _plans.popleft() if _plans else None
        return _active_plan


def cue_duration_ms(text: str, profile: str) -> int:
    """Estimate a safe visual hold from spoken length, without semantic rules."""

    compact_length = len(re.sub(r"\s+", "", str(text or "")))
    estimate = max(900, min(5200, 500 + compact_length * 150))
    if profile == "wink":
        return min(900, estimate)
    return estimate


def publish_expression(text: str, *_args, **_kwargs) -> ExpressionCue:
    """Bind the next LLM plan to TTS, or use a non-semantic neutral fallback."""

    global _sequence
    plan = _take_plan()
    with _lock:
        _sequence += 1
        cue = ExpressionCue(
            sequence=_sequence,
            profile=plan.profile if plan else "neutral",
            intensity=plan.intensity if plan else 0.0,
            duration_ms=cue_duration_ms(text, plan.profile if plan else "neutral"),
            mouth_strength=plan.mouth_strength if plan else 0.0,
            style=plan.style if plan else "neutral",
            source="llm" if plan else "fallback",
            created_at=time.monotonic(),
        )
        _cues.append(cue)
        return cue


def cues_after(sequence: int) -> list[ExpressionCue]:
    with _lock:
        return [cue for cue in _cues if cue.sequence > sequence]


def clear_delivery_state() -> None:
    """Discard unconsumed controls after a session interruption."""

    global _active_plan
    with _lock:
        _plans.clear()
        _active_plan = None


def begin_delivery_response() -> None:
    """Start a response while preserving the control queued by its first tag."""

    global _active_plan
    with _lock:
        _active_plan = None


__all__ = [
    "DELIVERY_CONTROL_PROMPT", "DeliveryControlFilter", "ExpressionCue",
    "begin_delivery_response", "clear_delivery_state", "cue_duration_ms", "cues_after",
    "publish_expression", "submit_delivery_plan",
]
