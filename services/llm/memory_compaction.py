"""Fast, deterministic memory extraction for live voice conversations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


SYNTHETIC_PROMPTS = (
    "start the conversation now",
    "现在主动欢迎刚连线的观众",
    "对方安静了一会儿",
)
STRUCTURED_PREFIX = "【结构化记忆】"
_PREFERENCE_RE = re.compile(r"不太喜欢|不喜欢|讨厌|不爱|最喜欢|很喜欢|偏爱|喜欢|爱")
_NEGATIVE_PREFERENCES = {"不太喜欢", "不喜欢", "讨厌", "不爱"}


def _clean(value: str, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", value).strip(" ，,。；;：:")[:limit]


def _dedupe_latest(values: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in reversed(values):
        clean = _clean(value)
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
        if len(result) >= limit:
            break
    result.reverse()
    return result


def _latest_identity(values: list[str]) -> list[str]:
    slots: dict[str, str] = {}
    other: list[str] = []
    for value in values:
        if any(word in value for word in ("我叫", "我现在叫", "我以前叫", "改名叫", "名字是", "叫我")):
            slots["name"] = value
        elif "住在" in value or value.startswith("我在"):
            slots["location"] = value
        elif "来自" in value:
            slots["origin"] = value
        elif "今年" in value:
            slots["age"] = value
        elif value.startswith("我是"):
            slots["identity"] = value
        else:
            other.append(value)
    return _dedupe_latest([*slots.values(), *other], 5)


def extract_transcript(messages: list[dict]) -> tuple[list[str], list[str]]:
    prompt = next(
        (
            str(message.get("content", ""))
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    match = re.search(
        r"---\s*CONVERSATION START\s*---(.*?)---\s*CONVERSATION END\s*---",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    transcript = match.group(1) if match else prompt
    user_parts: list[str] = []
    assistant_parts: list[str] = []
    current: list[str] | None = None

    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        role_match = re.match(r"^(User|Assistant):\s*(.*)$", line, re.IGNORECASE)
        if role_match:
            role, value = role_match.groups()
            current = user_parts if role.lower() == "user" else assistant_parts
            should_skip = role.lower() == "user" and any(
                marker in value.lower() for marker in SYNTHETIC_PROMPTS
            )
            if value and not should_skip:
                current.append(value)
            elif should_skip:
                current = None
            continue
        if line.startswith("[Tool "):
            assistant_parts.append(line)
            current = assistant_parts
        elif line and current:
            current[-1] = f"{current[-1]} {line}"
    return user_parts, assistant_parts


@dataclass
class UserMemory:
    identity: list[str] = field(default_factory=list)
    preferences: dict[str, tuple[bool, str]] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    def load_structured(self, text: str) -> bool:
        if STRUCTURED_PREFIX not in text:
            return False
        for name, value in re.findall(r"(?:^|｜)(身份|偏好|不喜欢|重要信息|近期话题)：([^｜]+)", text):
            items = [_clean(item) for item in re.split(r"[、；]", value) if _clean(item)]
            if name == "身份":
                self.identity.extend(items)
            elif name in ("偏好", "不喜欢"):
                positive = name == "偏好"
                for item in items:
                    self.preferences[item.casefold()] = (positive, item)
            elif name == "重要信息":
                self.facts.extend(items)
            else:
                self.topics.extend(items)
        return True

    def ingest(self, text: str) -> None:
        if self.load_structured(text):
            return
        clean = _clean(text)
        if not clean:
            return

        for pattern in (
            r"(?:我(?:现在|以前)?叫|我改名叫|我的名字是|叫我)([^，。！？；]{1,24})",
            r"(?:我来自|我是)([^，。！？；]{1,32})",
            r"(?:(?:我)?住在|我在)([^，。！？；]{1,32})",
            r"我今年([^，。！？；]{1,16})",
        ):
            self.identity.extend(match.group(0) for match in re.finditer(pattern, clean))

        matches = list(_PREFERENCE_RE.finditer(clean))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
            target = clean[match.end() : end]
            target = re.split(r"[，,。！？!?；]", target, maxsplit=1)[0]
            target = re.sub(r"(?:但|不过|可是)?(?:我)?(?:现在|已经|后来)?$", "", target)
            target = _clean(target, 48).rstrip("了")
            if target:
                self.preferences[target.casefold()] = (match.group(0) not in _NEGATIVE_PREFERENCES, target)

        clauses = [_clean(part) for part in re.split(r"[。！？!?；]", clean) if _clean(part)]
        for clause in clauses:
            is_structured = any(
                word in clause
                for word in (
                    "我叫",
                    "我现在叫",
                    "我以前叫",
                    "改名叫",
                    "名字是",
                    "叫我",
                    "我来自",
                    "我是",
                    "住在",
                    "我今年",
                    "喜欢",
                    "讨厌",
                    "不爱",
                )
            )
            if not is_structured and len(clause) >= 3:
                self.facts.append(clause)
            if "?" in text or "？" in text or any(word in clause for word in ("聊", "说", "问", "想知道", "怎么办")):
                self.topics.append(clause)

    def render(self, max_chars: int) -> str:
        positive = [value for state, value in self.preferences.values() if state]
        negative = [value for state, value in self.preferences.values() if not state]
        sections = [
            ("身份", _latest_identity(self.identity)),
            ("偏好", _dedupe_latest(positive, 8)),
            ("不喜欢", _dedupe_latest(negative, 8)),
            ("重要信息", _dedupe_latest(self.facts, 8)),
            ("近期话题", _dedupe_latest(self.topics, 5)),
        ]
        return _render_sections(sections, max_chars, "暂无需要长期保留的用户信息")


def _render_sections(sections: list[tuple[str, list[str]]], max_chars: int, fallback: str) -> str:
    output = STRUCTURED_PREFIX
    for name, values in sections:
        if not values:
            continue
        section = f"｜{name}：{'、'.join(values)}"
        remaining = max_chars - len(output)
        if remaining <= 8:
            break
        output += section[:remaining]
    return output if output != STRUCTURED_PREFIX else f"{STRUCTURED_PREFIX}｜{fallback}"


def build_local_memory(messages: list[dict], max_chars: int = 900) -> dict[str, str]:
    """Build structured summaries without model, network, disk, or GPU access."""
    user_parts, assistant_parts = extract_transcript(messages)
    memory = UserMemory()
    for part in user_parts:
        memory.ingest(part)

    commitments: list[str] = []
    recent: list[str] = []
    for part in assistant_parts:
        clean = _clean(part)
        if not clean:
            continue
        if STRUCTURED_PREFIX in clean:
            for name, value in re.findall(r"(?:^|｜)(承诺与结论|近期回复)：([^｜]+)", clean):
                (commitments if name == "承诺与结论" else recent).extend(
                    item for item in re.split(r"[、；]", value) if _clean(item)
                )
            continue
        recent.append(clean)
        if any(word in clean for word in ("记住", "答应", "下次", "会", "已经", "建议", "决定")):
            commitments.append(clean)

    assistant_summary = _render_sections(
        [
            ("承诺与结论", _dedupe_latest(commitments, 6)),
            ("近期回复", _dedupe_latest(recent, 6)),
        ],
        max_chars,
        "暂无需要长期保留的助手信息",
    )
    return {"user_summary": memory.render(max_chars), "assistant_summary": assistant_summary}


def local_compaction(messages: list[dict], max_chars: int = 900) -> str:
    return json.dumps(build_local_memory(messages, max_chars), ensure_ascii=False)
