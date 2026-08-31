"""Deterministic public-output boundary for model/provider protocols.

Only provider-native structured function-call objects are executable. Textual
reasoning or tool-call markup is private protocol noise and must never reach a
room transcript or TTS, regardless of which model produced it.
"""

from __future__ import annotations

import re


HIDDEN_PROTOCOL_TAGS = frozenset(
    {
        "think",
        "analysis",
        "reasoning",
        "tool_call",
        "toolcall",
        "function_call",
        "functioncall",
    }
)

# Presentation markup is never valid spoken/live-room text. Protocol blocks
# suppress their contents; formatting tags are removed while text is retained.
PUBLIC_BREAK_TAGS = frozenset({"br", "p", "div", "li", "ul", "ol", "blockquote", "pre"})
PUBLIC_INLINE_TAGS = frozenset(
    {"span", "strong", "em", "b", "i", "u", "s", "a", "code", "small", "mark"}
)


class PublicOutputFilter:
    """Incrementally remove private XML-like blocks split across any chunks."""

    def __init__(self) -> None:
        self.buffer = ""
        self.suppressed_depth = 0

    def feed(self, piece: str, *, final: bool = False) -> str:
        self.buffer += piece or ""
        output: list[str] = []
        while self.buffer:
            opening = self.buffer.find("<")
            if opening < 0:
                if not self.suppressed_depth:
                    output.append(self.buffer)
                self.buffer = ""
                break
            if opening > 0:
                if not self.suppressed_depth:
                    output.append(self.buffer[:opening])
                self.buffer = self.buffer[opening:]
            closing = self.buffer.find(">")
            if closing < 0:
                if final:
                    candidate = self.buffer[1:].strip().lower().lstrip("/")
                    candidate = re.split(r"\s+", candidate, maxsplit=1)[0]
                    hidden_prefix = any(tag.startswith(candidate) for tag in HIDDEN_PROTOCOL_TAGS)
                    if not self.suppressed_depth and not hidden_prefix:
                        output.append(self.buffer)
                    self.buffer = ""
                break
            raw_tag = self.buffer[: closing + 1]
            tag_body = self.buffer[1:closing].strip()
            is_closing = tag_body.startswith("/")
            tag_name = re.split(
                r"\s+", tag_body.lstrip("/").rstrip("/"), maxsplit=1
            )[0].lower()
            if tag_name in HIDDEN_PROTOCOL_TAGS:
                if is_closing:
                    self.suppressed_depth = max(0, self.suppressed_depth - 1)
                elif not tag_body.endswith("/"):
                    self.suppressed_depth += 1
            elif tag_name in PUBLIC_BREAK_TAGS:
                if not self.suppressed_depth and tag_name in {"br", "p", "div", "li"}:
                    output.append("\n")
            elif tag_name in PUBLIC_INLINE_TAGS:
                pass
            elif not self.suppressed_depth:
                output.append(raw_tag)
            self.buffer = self.buffer[closing + 1 :]
        return "".join(output)


def clean_public_output(text: str) -> str:
    cleaner = PublicOutputFilter()
    return cleaner.feed(str(text or ""), final=True).strip()


def contains_private_protocol(text: str) -> bool:
    """Detect complete or partial private tags after a candidate is generated."""
    lowered = str(text or "").lower()
    return any(re.search(rf"</?\s*{re.escape(tag)}\b", lowered) for tag in HIDDEN_PROTOCOL_TAGS)
