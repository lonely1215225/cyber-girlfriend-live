#!/usr/bin/env python3
"""OpenAI /v1/responses shim in front of Ollama /api/chat with think=false.

Ollama's /v1/responses ignores think=false and spends 20s+ on hidden reasoning.
/api/chat?think=false replies in ~1s. speech-to-speech talks to this shim.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_WEB_DIR = Path(__file__).resolve().parents[2] / "apps" / "web"
if str(_WEB_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_DIR))

from dialogue_intent import (
    SPOKEN_CHINESE_POLICY,
    needs_web_search,
    viewer_is_chinese,
    viewer_utterance,
)
from memory_compaction import local_compaction as _local_compaction
from output_harness import PublicOutputFilter, clean_public_output

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
HOST = os.environ.get("THINKLESS_HOST", "127.0.0.1")
PORT = int(os.environ.get("THINKLESS_PORT", "11435"))
KEEP_ALIVE = os.environ.get("LLM_KEEP_ALIVE", "-1").strip() or "-1"
PREWARM = os.environ.get("LLM_PREWARM", "1").strip().lower() in {"1", "true", "yes", "on"}
COMPACTION_NUM_PREDICT = int(os.environ.get("LLM_COMPACTION_NUM_PREDICT", "256"))
WELCOME_NUM_PREDICT = max(96, int(os.environ.get("LLM_WELCOME_NUM_PREDICT", "128")))
WELCOME_RETRY_NUM_PREDICT = max(
    WELCOME_NUM_PREDICT,
    int(os.environ.get("LLM_WELCOME_RETRY_NUM_PREDICT", "192")),
)
NEWS_NUM_PREDICT = max(160, int(os.environ.get("LLM_NEWS_NUM_PREDICT", "256")))
NEWS_CONTINUE_NUM_PREDICT = max(
    64, int(os.environ.get("LLM_NEWS_CONTINUE_NUM_PREDICT", "128"))
)
NEWS_RETRY_NUM_PREDICT = max(
    NEWS_NUM_PREDICT,
    int(os.environ.get("LLM_NEWS_RETRY_NUM_PREDICT", "256")),
)
DIALOGUE_CONTINUE_NUM_PREDICT = max(
    48, int(os.environ.get("LLM_DIALOGUE_CONTINUE_NUM_PREDICT", "128"))
)
COMPACTION_MODE = os.environ.get("LLM_COMPACTION_MODE", "local").strip().lower()
COMPACTION_MAX_CHARS = max(300, int(os.environ.get("LLM_COMPACTION_MAX_CHARS", "900")))
GROK_ENABLED = os.environ.get("GROK_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
GROK_BASE_URL = os.environ.get("GROK_PROXY_BASE_URL", "http://127.0.0.1:18080/v1").rstrip("/")
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-4.6").strip() or "grok-4.6"
GROK_FAST_MODEL = os.environ.get("GROK_FAST_MODEL", "grok-4.5").strip() or "grok-4.5"
GROK_API_KEY = os.environ.get("GROK_PROXY_API_KEY", "").strip()
GROK_REASONING_EFFORT = os.environ.get("GROK_REASONING_EFFORT", "low").strip().lower() or "low"
GROK_TIMEOUT_SECONDS = max(2.0, float(os.environ.get("GROK_TIMEOUT_SECONDS", "45")))
LOCAL_READ_TIMEOUT_SECONDS = max(
    4.0, float(os.environ.get("LLM_LOCAL_READ_TIMEOUT_SECONDS", "12.0"))
)
BUFFERED_READ_TIMEOUT_SECONDS = max(
    LOCAL_READ_TIMEOUT_SECONDS,
    float(os.environ.get("LLM_BUFFERED_READ_TIMEOUT_SECONDS", "45")),
)
LOCAL_TIMEOUT_REPLY = "嗯？刚才没接稳，你再说一遍嘛。"
LOCAL_CONVERSATION_NUM_PREDICT = max(
    80, int(os.environ.get("LLM_LOCAL_CONVERSATION_NUM_PREDICT", "160"))
)
SIMPLE_CHAT_NUM_PREDICT = max(
    64, int(os.environ.get("LLM_SIMPLE_CHAT_NUM_PREDICT", "96"))
)
LOCAL_LEAD_ENABLED = os.environ.get("LOCAL_LEAD_ENABLED", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
LOCAL_LEAD_MODEL = os.environ.get(
    "LOCAL_LEAD_MODEL", "jaahas/qwen3.5-uncensored:9b"
).strip()
LOCAL_LEAD_TIMEOUT_SECONDS = max(
    0.25, float(os.environ.get("LOCAL_LEAD_TIMEOUT_SECONDS", "1.4"))
)
LOCAL_LEAD_MAX_CHARS = max(8, int(os.environ.get("LOCAL_LEAD_MAX_CHARS", "24")))
LEAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="local-lead")
GROK_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="grok-upstream")


ModelOutputSanitizer = PublicOutputFilter


def clean_model_output(text: str) -> str:
    return clean_public_output(text)


def _keep_alive_value() -> int | str:
    """Ollama accepts durations as strings, but sentinel values as integers."""
    try:
        return int(KEEP_ALIVE)
    except ValueError:
        return KEEP_ALIVE


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                parts.append(c.get("text") or c.get("input_text") or "")
        return "".join(parts)
    return ""


def to_messages(inp) -> list[dict]:
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}]
    msgs = []
    for item in inp or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            arguments = item.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"input": arguments}
            msgs.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id") or "",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            )
            continue
        if item_type == "function_call_output":
            output = item.get("output") or ""
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            msgs.append({"role": "tool", "content": output})
            continue
        role = item.get("role") or "user"
        if role not in ("system", "user", "assistant"):
            role = "user"
        text = _extract_text(item.get("content"))
        if text.strip():
            msgs.append({"role": role, "content": text})
    return msgs or [{"role": "user", "content": "你好"}]


def request_messages(payload: dict) -> list[dict]:
    """Preserve Responses API top-level instructions as an Ollama system turn."""

    messages = to_messages(payload.get("input"))
    instructions = _extract_text(payload.get("instructions")).strip()
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})
    return messages


def to_ollama_tools(tools) -> list[dict]:
    """Translate flat Responses API function declarations to Ollama chat tools."""
    result = []
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        name = tool.get("name")
        if not name:
            continue
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return result


def _is_compaction_request(messages: list[dict]) -> bool:
    return any(
        "conversation memory compressor" in str(message.get("content", "")).lower()
        for message in messages
    )


def _is_exact_speech_request(messages: list[dict]) -> bool:
    """Fixed TTS readouts do not need a remote reasoning-model request."""
    return any(
        "只逐字朗读用户提供的文字" in str(message.get("content", ""))
        for message in messages
    )


def _is_room_welcome_request(messages: list[dict]) -> bool:
    """Recognize the server-owned arrival greeting workflow."""
    return any(
        "这是直播间入场欢迎" in content or "直播间入场欢迎生成器" in content
        for message in messages
        for content in [str(message.get("content", ""))]
    )


def _is_proactive_broadcast_request(messages: list[dict]) -> bool:
    """Recognize server-owned unattended broadcasts, never user semantics."""
    return any(
        "无人连线时的直播间主动播报" in str(message.get("content", ""))
        for message in messages
    )


_SENTENCE_END_CHARS = frozenset("。！？!?～~…")
_DELIVERY_TAG_RE = re.compile(r"<e\b[^>]*>", re.IGNORECASE)
_DELIVERY_TAG_OPEN_RE = re.compile(r"<e\b[^>]*$", re.IGNORECASE)


def _visible_spoken_text(text: str) -> str:
    """Spoken prose after stripping protocol and hidden delivery tags."""
    visible = clean_model_output(text)
    visible = _DELIVERY_TAG_RE.sub("", visible)
    visible = _DELIVERY_TAG_OPEN_RE.sub("", visible)
    return visible.strip()


def _spoken_text_is_incomplete(text: str) -> bool:
    """Whether visible speech stops mid-clause instead of at a sentence end."""
    visible = _visible_spoken_text(text)
    if not visible:
        return False
    visible = visible.rstrip("\"'”’）)】」 \t")
    return bool(visible) and visible[-1] not in _SENTENCE_END_CHARS


def _should_finish_incomplete(done_reason: str, text: str, tool_calls: list | None) -> bool:
    if tool_calls:
        return False
    if not _visible_spoken_text(text):
        return False
    return done_reason == "length" or _spoken_text_is_incomplete(text)


def _shorter_complete_retry_messages(
    messages: list[dict], raw_text: str, kind: str
) -> list[dict]:
    if kind == "welcome":
        hint = (
            "上一句因长度限制没有说完。请重新生成一句更短、语义完整的欢迎词；"
            "仍遵守原要求，并用句号、问号、感叹号或波浪号自然收尾。"
        )
    elif kind == "news":
        hint = (
            "上一段因长度限制没有说完。请重新生成更短、语义完整的播报；"
            "两到三句讲清事实，再加一句邀请，必须用句号、问号或感叹号收尾，不要半句。"
        )
    else:
        hint = (
            "上一段因长度限制没有说完。请重新生成更短、语义完整的回复；"
            "仍遵守原要求，必须用句号、问号或感叹号收尾。"
        )
    return [
        *messages,
        {"role": "assistant", "content": raw_text},
        {"role": "user", "content": hint},
    ]


def _continuation_messages(messages: list[dict], raw_text: str) -> list[dict]:
    return [
        *messages,
        {"role": "assistant", "content": raw_text},
        {
            "role": "user",
            "content": (
                "上一段因长度限制在半句处停住了。从断开的最后一个字后面接着写完，"
                "不要重复已经写出的内容，写完用句号、问号或感叹号收尾。"
            ),
        },
    ]


def _ollama_chat_json(
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    *,
    num_predict: int | None = None,
    timeout_s: float | None = None,
) -> dict:
    with ollama_chat(
        model,
        messages,
        False,
        tools,
        num_predict_override=num_predict,
        timeout_s=timeout_s,
    ) as response:
        return json.loads(response.read())


def _finish_buffered_local_chat(
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    *,
    kind: str,
    first_limit: int,
    retry_limit: int,
    continue_limit: int | None = None,
) -> tuple[dict, int, int]:
    """Generate one complete turn before any token is exposed downstream."""
    timeout_s = BUFFERED_READ_TIMEOUT_SECONDS
    data = _ollama_chat_json(
        model, messages, tools, num_predict=first_limit, timeout_s=timeout_s
    )
    message = data.get("message") or {}
    raw_text = str(message.get("content") or "")
    done_reason = str(data.get("done_reason") or "")
    tool_calls = message.get("tool_calls") or []
    in_tok = int(data.get("prompt_eval_count") or 0)
    out_tok = int(data.get("eval_count") or 0)

    if continue_limit and _should_finish_incomplete(done_reason, raw_text, tool_calls):
        print(
            f"[thinkless] {kind} incomplete (reason={done_reason or 'stop'}); continuing",
            flush=True,
        )
        continued = _ollama_chat_json(
            model,
            _continuation_messages(messages, raw_text),
            tools,
            num_predict=continue_limit,
            timeout_s=timeout_s,
        )
        extra = str((continued.get("message") or {}).get("content") or "")
        raw_text = f"{raw_text}{extra}"
        done_reason = str(continued.get("done_reason") or "")
        in_tok += int(continued.get("prompt_eval_count") or 0)
        out_tok += int(continued.get("eval_count") or 0)
        message = dict(message)
        message["content"] = raw_text
        data = dict(data)
        data["message"] = message
        data["done_reason"] = done_reason

    if _should_finish_incomplete(done_reason, raw_text, tool_calls):
        print(
            f"[thinkless] {kind} still incomplete; regenerating a shorter complete line",
            flush=True,
        )
        data = _ollama_chat_json(
            model,
            _shorter_complete_retry_messages(messages, raw_text, kind),
            tools,
            num_predict=retry_limit,
            timeout_s=timeout_s,
        )
        message = data.get("message") or {}
        raw_text = str(message.get("content") or "")
        in_tok += int(data.get("prompt_eval_count") or 0)
        out_tok += int(data.get("eval_count") or 0)
    return data, in_tok, out_tok


def _is_public_comment_request(messages: list[dict]) -> bool:
    """Route room comments to the stronger conversational model.

    This marker is server-owned and therefore safe for structural routing. It
    does not infer intent from viewer wording or encode answer semantics.
    """

    return any(
        message.get("role") == "system"
        and "这是公开评论" in str(message.get("content", ""))
        for message in messages
    )


_CHAT_ONLY_TOOL_NAMES = {
    frozenset({"request_external_capabilities"}),
    frozenset({"smart_web_search"}),
}


def _is_fast_discovery_turn(payload: dict) -> bool:
    """Use the resident model for the first chat turn.

    Ordinary conversation is answered in this same turn after tools are
    stripped. If the viewer actually asked for a live fact, the same hop
    keeps ``smart_web_search`` and forces that one call.
    """
    tools = [tool for tool in payload.get("tools") or [] if isinstance(tool, dict)]
    names = frozenset(str(tool.get("name") or "") for tool in tools)
    if names not in _CHAT_ONLY_TOOL_NAMES:
        return False
    return not any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in payload.get("input") or []
    )


def _needs_reliable_external_route(messages: list[dict]) -> bool:
    """Recognize explicit freshness/lookup requests that must not be hallucinated.

    Intent is taken from the viewer's current line only. Packed memory or a
    leftover news wrapper must not turn a joke request into a live search.
    """
    current = next(
        (
            str(item.get("content") or "")
            for item in reversed(messages)
            if item.get("role") == "user"
        ),
        "",
    )
    return needs_web_search(viewer_utterance(current))


def _is_fast_conversation_followup(payload: dict) -> bool:
    """Keep the post-router ordinary reply local; research still uses Grok."""
    if payload.get("tools"):
        return False
    for item in reversed(payload.get("input") or []):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        output = item.get("output")
        if not isinstance(output, str):
            output = json.dumps(output or {}, ensure_ascii=False)
        return '"route": "conversation_fast"' in output or '"route":"conversation_fast"' in output
    return False


def _is_fast_external_planning(payload: dict) -> bool:
    """Use the local model for the first concrete tool selection.

    The browser plays its single progress sentence while the tool runs, and the
    remote reasoning model is still used after evidence returns. Keeping this
    bounded planning hop local avoids a network/model round trip before search.
    """
    tools = [tool for tool in payload.get("tools") or [] if isinstance(tool, dict)]
    if not tools or {str(tool.get("name") or "") for tool in tools} == {"request_external_capabilities"}:
        return False
    for item in reversed(payload.get("input") or []):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        output = item.get("output")
        if not isinstance(output, str):
            output = json.dumps(output or {}, ensure_ascii=False)
        return '"route": "external_research"' in output or '"route":"external_research"' in output
    return False


def local_compaction(messages: list[dict]) -> str:
    """Compatibility wrapper used by tests and the Responses API handler."""
    return _local_compaction(messages, COMPACTION_MAX_CHARS)


def _num_predict_for_messages(messages: list[dict]) -> int:
    """Reserve enough output for both the hidden delivery plan and spoken text."""
    num_predict = int(os.environ.get("LLM_NUM_PREDICT", "256"))
    if _is_compaction_request(messages):
        return COMPACTION_NUM_PREDICT
    if _is_room_welcome_request(messages):
        return WELCOME_NUM_PREDICT
    if _is_proactive_broadcast_request(messages):
        return NEWS_NUM_PREDICT
    return num_predict


def ollama_chat(
    model: str,
    messages: list[dict],
    stream: bool,
    tools: list[dict] | None = None,
    *,
    num_predict_override: int | None = None,
    timeout_s: float | None = None,
):
    # Default Ollama ctx for this 9B is 262144. Prompt eval then takes ~8s even
    # for a short line, and under GPU contention it exceeds the s2s 20s timeout.
    num_ctx = int(os.environ.get("LLM_NUM_CTX", "4096"))
    num_predict = (
        max(1, int(num_predict_override))
        if num_predict_override is not None
        else _num_predict_for_messages(messages)
    )
    read_timeout = (
        LOCAL_READ_TIMEOUT_SECONDS
        if timeout_s is None
        else max(LOCAL_READ_TIMEOUT_SECONDS, float(timeout_s))
    )
    payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": False,
            # The first request after Ollama unloads a model was taking 6-8s.
            # Keep the conversation model resident; this box is dedicated to
            # the live pipeline and has enough VRAM for the configured model.
            "keep_alive": _keep_alive_value(),
            "options": {"num_ctx": num_ctx, "num_predict": num_predict},
        }
    if tools:
        payload["tools"] = tools
    body = json.dumps(payload).encode()
    req = Request(
        f"{OLLAMA}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # urllib applies this as a socket inactivity timeout. A healthy streaming
    # model keeps producing data; a wedged GPU/request is cut off before the
    # realtime pipeline's much slower 20-second outer timeout.
    return urlopen(req, timeout=read_timeout)


def grok_response(
    payload: dict, *, model: str | None = None, reasoning_effort: str | None = None
):
    """Forward one Responses API request to the private Grok OAuth proxy.

    The public application continues to address this shim with its logical
    model name.  Provider-specific model rewriting happens only here, so the
    same request can safely fall back to the local Ollama model.
    """
    forwarded = dict(payload)
    forwarded["model"] = model or GROK_MODEL
    forwarded["reasoning"] = {
        "effort": reasoning_effort or GROK_REASONING_EFFORT
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if forwarded.get("stream") else "application/json"}
    if GROK_API_KEY:
        headers["Authorization"] = f"Bearer {GROK_API_KEY}"
    request = Request(
        f"{GROK_BASE_URL}/responses",
        data=json.dumps(forwarded, ensure_ascii=False).encode(),
        headers=headers,
        method="POST",
    )
    return urlopen(request, timeout=GROK_TIMEOUT_SECONDS)


def local_conversation_lead(messages: list[dict]) -> str:
    """Generate one fact-free spoken bridge while Grok works in parallel.

    This model never owns the answer. A deterministic safety gate rejects
    numbers, URLs and protocol-like output so a fast lead cannot race an
    evidence-backed result with a made-up fact.
    """
    current = next(
        (
            str(item.get("content") or "")
            for item in reversed(messages)
            if item.get("role") == "user"
        ),
        "",
    ).split("\n\n【", 1)[0].strip()
    if not current:
        return ""
    lead_messages = [
        {
            "role": "system",
            "content": (
                "你是小麻，只生成实时对话开头的一句自然接话。"
                "先接住对方刚说的话，短、直、像随口说，可以带一点点磕绊。"
                "不要“既然……那就……”这类书面腔。"
                "这句只负责承接，不回答事实，不给价格、日期、新闻结论，不承诺查询，"
                "不复述问题，不说正在思考，不添加对方没提过的具体物品、经历或动作。"
                "不能讽刺、嫌弃、贬低或指导投资。对事实问题只表达认真对待。"
                "只输出一句八到二十四个汉字的完整中文口语。"
            ),
        },
        {"role": "user", "content": current[:500]},
    ]
    with ollama_chat(
        LOCAL_LEAD_MODEL,
        lead_messages,
        False,
        [],
        num_predict_override=40,
    ) as response:
        data = json.loads(response.read())
    text = clean_model_output(str((data.get("message") or {}).get("content") or ""))
    text = re.sub(r"\s+", "", text).strip()
    match = re.search(r"[。！？!?～~]", text)
    if match:
        text = text[: match.end()]
    if len(text) > LOCAL_LEAD_MAX_CHARS:
        text = text[:LOCAL_LEAD_MAX_CHARS].rstrip("，、；：") + "。"
    elif text and text[-1] not in "。！？!?～~":
        text += "。"
    if (
        len(re.findall(r"[\u3400-\u9fff]", text)) < 6
        or re.search(r"\d|https?://|<[^>]+>|(?:tool|function)_call|[¥￥$€]", text, re.I)
    ):
        return ""
    return text


def response_output_text(payload: dict) -> str:
    """Derive the SDK-style output_text helper from a raw Responses result."""
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return clean_model_output(direct)
    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return clean_model_output("".join(parts))


def sanitize_response_payload(payload: dict, *, visible_text: str | None = None) -> dict:
    """Apply the public boundary to every textual view of a Responses result."""
    if not isinstance(payload, dict):
        return payload
    override = clean_model_output(visible_text) if visible_text is not None else None
    if isinstance(payload.get("output_text"), str):
        payload["output_text"] = override if override is not None else clean_model_output(payload["output_text"])
    if isinstance(payload.get("output"), list):
        payload["output"] = [
            item for item in payload["output"]
            if not isinstance(item, dict) or item.get("type") != "reasoning"
        ]
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                content["text"] = (
                    override
                    if override is not None
                    else clean_model_output(str(content.get("text") or ""))
                )
    return payload


def _sanitize_stream_event(
    payload: dict,
    cleaner: PublicOutputFilter,
    visible: list[str],
) -> tuple[dict | None, str]:
    """Sanitize one Grok SSE event while retaining structured function calls."""
    event_type = str(payload.get("type") or "")
    tail = ""
    if event_type == "response.output_text.delta":
        delta = cleaner.feed(str(payload.get("delta") or ""))
        if not delta:
            return None, ""
        visible.append(delta)
        payload["delta"] = delta
    elif event_type in {"response.output_text.done", "response.completed", "response.done"}:
        tail = cleaner.feed("", final=True)
        if tail:
            visible.append(tail)
        text = "".join(visible)
        if event_type == "response.output_text.done":
            payload["text"] = text
        response = payload.get("response")
        if isinstance(response, dict):
            sanitize_response_payload(response, visible_text=text)
    elif event_type == "response.output_item.done":
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "message":
            sanitize_response_payload({"output": [item]}, visible_text="".join(visible))
    elif event_type in {"response.content_part.added", "response.content_part.done"}:
        part = payload.get("part")
        if isinstance(part, dict) and part.get("type") == "output_text":
            part["text"] = "".join(visible)
    elif "reasoning" in event_type:
        # Reasoning summaries are provider diagnostics, not public response
        # content. The client does not need them to execute structured tools.
        return None, ""
    return payload, tail


def relay_sanitized_grok_stream(
    upstream, writer, *, prefix_text: str = "", eager_prefix: bool = False
) -> None:
    """Relay provider SSE without ever forwarding private protocol text."""
    cleaner = PublicOutputFilter()
    visible: list[str] = []
    block: list[str] = []
    prefix_item_id = "msg_lead_" + uuid.uuid4().hex[:16]
    prefix_injected = False
    synthetic_response_id = "resp_hybrid_" + uuid.uuid4().hex[:16]

    def write_event(event_type: str, payload: dict) -> None:
        writer(
            (
                f"event: {event_type}\n"
                f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            ).encode()
        )

    def inject_prefix() -> None:
        nonlocal prefix_injected
        if prefix_injected or not prefix_text:
            return
        prefix_injected = True
        item = {
            "id": prefix_item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        write_event(
            "response.output_item.added",
            {"type": "response.output_item.added", "output_index": 0, "item": item},
        )
        write_event(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": prefix_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )
        write_event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": prefix_item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": prefix_text,
            },
        )
        write_event(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": prefix_item_id,
                "output_index": 0,
                "content_index": 0,
                "text": prefix_text,
            },
        )
        completed_item = {
            **item,
            "status": "completed",
            "content": [{"type": "output_text", "text": prefix_text, "annotations": []}],
        }
        write_event(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": prefix_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": completed_item["content"][0],
            },
        )
        write_event(
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": 0, "item": completed_item},
        )

    grok_started = time.monotonic()
    first_visible_at: float | None = None

    if eager_prefix and prefix_text:
        write_event(
            "response.created",
            {
                "type": "response.created",
                "response": {"id": synthetic_response_id, "status": "in_progress"},
            },
        )
        inject_prefix()
        first_visible_at = time.monotonic()

    def emit(lines: list[str]) -> None:
        nonlocal first_visible_at
        if not lines:
            return
        data_lines = [line[5:].lstrip() for line in lines if line.startswith("data:")]
        if not data_lines or data_lines == ["[DONE]"]:
            writer(("\n".join(lines) + "\n\n").encode())
            return
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            # Unknown provider framing is withheld rather than risking public
            # protocol leakage. Structured Responses events are JSON.
            return
        sanitized, tail = _sanitize_stream_event(payload, cleaner, visible)
        if first_visible_at is None and (
            tail or (sanitized and sanitized.get("type") == "response.output_text.delta")
        ):
            first_visible_at = time.monotonic()
        if tail:
            synthetic = {
                "type": "response.output_text.delta",
                "delta": tail,
                "content_index": 0,
                "output_index": 0,
            }
            writer(
                (
                    "event: response.output_text.delta\n"
                    f"data: {json.dumps(synthetic, ensure_ascii=False)}\n\n"
                ).encode()
            )
        if sanitized is None:
            return
        event_type = str(sanitized.get("type") or "")
        if prefix_text and event_type == "response.created" and eager_prefix:
            return
        if prefix_text and event_type == "response.created":
            event_lines = [line for line in lines if not line.startswith("data:")]
            event_lines.append(f"data: {json.dumps(sanitized, ensure_ascii=False)}")
            writer(("\n".join(event_lines) + "\n\n").encode())
            inject_prefix()
            return
        if prefix_text and "output_index" in sanitized:
            try:
                sanitized["output_index"] = int(sanitized["output_index"]) + 1
            except (TypeError, ValueError):
                pass
        if prefix_text and event_type == "response.output_text.done":
            sanitized["text"] = prefix_text + str(sanitized.get("text") or "")
        response = sanitized.get("response")
        if prefix_text and isinstance(response, dict) and event_type in {
            "response.completed", "response.done"
        }:
            if eager_prefix:
                response["id"] = synthetic_response_id
            prefix_item = {
                "id": prefix_item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": prefix_text, "annotations": []}],
            }
            response["output"] = [prefix_item, *(response.get("output") or [])]
            response["output_text"] = prefix_text + "".join(visible)
        event_lines = [line for line in lines if not line.startswith("data:")]
        event_lines.append(f"data: {json.dumps(sanitized, ensure_ascii=False)}")
        writer(("\n".join(event_lines) + "\n\n").encode())

    for raw in upstream:
        line = raw.decode("utf-8", "ignore").rstrip("\r\n")
        if line:
            block.append(line)
        else:
            emit(block)
            block = []
    emit(block)
    first_ms = (
        (first_visible_at - grok_started) * 1000.0
        if first_visible_at is not None else -1.0
    )
    print(
        f"[thinkless] grok latency first={first_ms:.0f}ms "
        f"total={(time.monotonic() - grok_started) * 1000:.0f}ms "
        f"chars={len(''.join(visible))}",
        flush=True,
    )


def _function_call_item(call: dict) -> dict:
    function = call.get("function") or {}
    arguments = function.get("arguments") or {}
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": "fc_" + uuid.uuid4().hex[:20],
        "type": "function_call",
        "call_id": call.get("id") or "call_" + uuid.uuid4().hex[:20],
        "name": function.get("name") or call.get("name") or "",
        "arguments": arguments,
        "status": "completed",
    }


def completed_response(
    model: str,
    text: str,
    in_tok: int,
    out_tok: int,
    tool_calls: list[dict] | None = None,
) -> dict:
    rid = "resp_" + uuid.uuid4().hex[:20]
    mid = "msg_" + uuid.uuid4().hex[:20]
    output = []
    if text or not tool_calls:
        output.append(
            {
                "id": mid,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    output.extend(_function_call_item(call) for call in (tool_calls or []))
    return {
        "id": rid,
        "object": "response",
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": text,
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("[thinkless]", self.address_string(), fmt % args, flush=True)

    def _send(self, code: int, payload: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/v1/models", "/models"):
            with urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
                data = json.loads(r.read())
            models = [
                {"id": m.get("name"), "object": "model"}
                for m in data.get("models") or []
                if m.get("name")
            ]
            if GROK_ENABLED:
                models.insert(0, {"id": GROK_MODEL, "object": "model", "owned_by": "xai"})
            body = json.dumps({"object": "list", "data": models}).encode()
            self._send(200, body, "application/json")
            return
        self._send(404, b'{"error":"not found"}', "application/json")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/responses", "/responses"):
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        model = req.get("model") or os.environ.get("LLM_NAME", "llama3")
        messages = request_messages(req)
        tools = to_ollama_tools(req.get("tools"))
        want_stream = bool(req.get("stream"))
        fast_discovery = _is_fast_discovery_turn(req)
        fast_conversation = _is_fast_conversation_followup(req)
        fast_external_planning = _is_fast_external_planning(req)
        fast_welcome = _is_room_welcome_request(messages)
        proactive_broadcast = _is_proactive_broadcast_request(messages)
        public_comment = _is_public_comment_request(messages)
        looking_up = _needs_reliable_external_route(messages)
        explicit_external_request = fast_discovery and looking_up
        has_tool_evidence = any(
            isinstance(item, dict) and item.get("type") == "function_call_output"
            for item in req.get("input") or []
        )
        ordinary_no_tool_turn = (
            not req.get("tools")
            and not has_tool_evidence
            and not proactive_broadcast
            and not _is_exact_speech_request(messages)
        )
        simple_chat = (
            not looking_up
            and not has_tool_evidence
            and not proactive_broadcast
            and not fast_welcome
            and not _is_exact_speech_request(messages)
            and (public_comment or ordinary_no_tool_turn or fast_discovery)
        )
        # One request must have exactly one prose generator.  The previous
        # hybrid path exposed a local answer and then appended a second Grok
        # answer, which sounded like repeated replies and delayed newer room
        # comments.  The resident model now owns bounded low-latency turns;
        # Grok owns evidence-backed synthesis after real tool output.
        # Companion and public-comment turns have no tools. Sending them to
        # Grok hid the first spoken token behind a 5-13s provider handshake.
        # Keep Grok for evidence-backed research and unattended news only.
        local_primary = (
            fast_discovery
            or fast_conversation
            or fast_external_planning
            or fast_welcome
            or ordinary_no_tool_turn
            or public_comment
        )
        if explicit_external_request:
            current = next(
                (str(item.get("content") or "") for item in reversed(messages) if item.get("role") == "user"),
                "",
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Live lookup only. Call smart_web_search exactly once with a concrete query. "
                        "Do not answer, estimate, mention a tool, or claim a result before verified "
                        "tool output."
                    ),
                },
                {"role": "user", "content": viewer_utterance(current)},
            ]
        elif simple_chat or fast_discovery:
            # Casual chat never sees a tool schema. Intent is decided here,
            # not by hoping a small model will ignore web search.
            tools = []
        elif fast_external_planning:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "Fast tool planning only. Call the best available tool immediately. "
                        "Do not answer or claim a result before tool output."
                    ),
                },
            )
        if (
            not _is_exact_speech_request(messages)
            and not proactive_broadcast
            and not explicit_external_request
            and (public_comment or viewer_is_chinese(
                next(
                    (str(item.get("content") or "") for item in reversed(messages) if item.get("role") == "user"),
                    "",
                )
            ))
        ):
            messages.insert(0, {"role": "system", "content": SPOKEN_CHINESE_POLICY})

        if COMPACTION_MODE == "local" and _is_compaction_request(messages):
            text = local_compaction(messages)
            out = completed_response(model, text, 0, 0)
            self._send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json")
            return

        # Local Ollama owns bounded low-latency turns: ordinary chat, welcome,
        # public comments, capability discovery, and first tool planning.
        # Grok owns evidence-backed synthesis after real tool output, plus
        # unattended news when enabled. Public-output safety is enforced below.
        if (
            GROK_ENABLED
            and not _is_exact_speech_request(messages)
            and not local_primary
        ):
            try:
                # Structural workflow routing, not keyword answer logic:
                # ordinary conversation and capability selection need low TTFT,
                # while evidence-backed synthesis keeps the strongest model.
                # No local 9B prose is exposed on either path.
                use_fast_grok = (
                    fast_discovery
                    or fast_conversation
                    or fast_welcome
                    or (not tools and not has_tool_evidence)
                )
                route_model = GROK_FAST_MODEL if use_fast_grok else GROK_MODEL
                route_effort = "none" if use_fast_grok else GROK_REASONING_EFFORT
                # Hybrid prose is deliberately disabled.  Running two models
                # is useful for background work, but only one of them may own
                # the user-visible answer for a turn.
                use_local_lead = False
                lead_started = time.monotonic()
                lead_future = (
                    LEAD_POOL.submit(local_conversation_lead, messages)
                    if use_local_lead
                    else None
                )
                grok_request = dict(req)
                grok_request["instructions"] = (
                    f"{str(req.get('instructions') or '').strip()}\n{SPOKEN_CHINESE_POLICY}"
                ).strip()
                if use_local_lead:
                    grok_request["instructions"] += (
                        "\n同一轮会先播放一句本地生成的简短接话，它已经负责接住情绪、"
                        "态度和第一层直接回应。请从补充信息、解释或推进话题开始，"
                        "不要再次问候、安慰、复述问题或重复第一层结论。"
                    )
                print(
                    f"[thinkless] route provider=grok model={route_model} "
                    f"effort={route_effort} stream={want_stream} "
                    f"discovery={fast_discovery} planning={fast_external_planning} "
                    f"welcome={fast_welcome}",
                    flush=True,
                )
                grok_future = (
                    GROK_POOL.submit(
                        grok_response,
                        grok_request,
                        model=route_model,
                        reasoning_effort=route_effort,
                    )
                    if use_local_lead
                    else None
                )
                # Hybrid streaming must expose HTTP immediately; waiting for
                # Grok's response headers here previously hid a ready 1-second
                # local lead behind a 5-13 second provider handshake.
                if use_local_lead:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                lead_text = ""
                if lead_future is not None:
                    remaining = max(
                        0.0,
                        LOCAL_LEAD_TIMEOUT_SECONDS - (time.monotonic() - lead_started),
                    )
                    try:
                        lead_text = lead_future.result(timeout=remaining)
                    except FutureTimeout:
                        lead_future.cancel()
                        print("[thinkless] local lead missed deadline; Grok continues", flush=True)
                    except Exception as exc:
                        print(
                            f"[thinkless] local lead skipped: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                if grok_future is not None:
                    try:
                        upstream = grok_future.result(timeout=GROK_TIMEOUT_SECONDS)
                    except Exception as exc:
                        print(
                            f"[thinkless] Grok hybrid stream failed: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        return
                    with upstream:
                        try:
                            relay_sanitized_grok_stream(
                                upstream,
                                lambda chunk: (
                                    self.wfile.write(chunk), self.wfile.flush()
                                ),
                                prefix_text=lead_text,
                                eager_prefix=bool(lead_text),
                            )
                        except (OSError, ValueError, json.JSONDecodeError) as exc:
                            print(
                                f"[thinkless] Grok hybrid stream ended early: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                    return
                upstream = grok_response(
                    grok_request, model=route_model, reasoning_effort=route_effort
                )
                if not want_stream:
                    with upstream:
                        data = json.loads(upstream.read())
                    data["output_text"] = response_output_text(data)
                    sanitize_response_payload(data)
                    has_tool_call = any(
                        isinstance(item, dict) and item.get("type") == "function_call"
                        for item in data.get("output") or []
                    )
                    if data.get("status") != "completed" or not (data["output_text"] or has_tool_call):
                        raise RuntimeError("Grok returned no completed answer")
                    self._send(200, json.dumps(data, ensure_ascii=False).encode(), "application/json")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with upstream:
                    try:
                        relay_sanitized_grok_stream(
                            upstream,
                            lambda chunk: (self.wfile.write(chunk), self.wfile.flush()),
                            prefix_text=lead_text,
                        )
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        # Headers may already be visible to the realtime client;
                        # never splice a second provider into the same SSE body.
                        print(
                            f"[thinkless] Grok stream ended early: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                return
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
                print(f"[thinkless] Grok unavailable, using local fallback: {type(exc).__name__}: {exc}", flush=True)

        if not want_stream:
            reason = "fast_primary" if local_primary else "fallback_or_exact"
            print(f"[thinkless] route provider=ollama reason={reason}", flush=True)
            started = time.monotonic()
            try:
                if fast_welcome or proactive_broadcast:
                    data, _, _ = _finish_buffered_local_chat(
                        model,
                        messages,
                        tools,
                        kind="welcome" if fast_welcome else "news",
                        first_limit=(
                            WELCOME_NUM_PREDICT if fast_welcome else NEWS_NUM_PREDICT
                        ),
                        retry_limit=(
                            WELCOME_RETRY_NUM_PREDICT
                            if fast_welcome
                            else NEWS_RETRY_NUM_PREDICT
                        ),
                        continue_limit=(
                            None if fast_welcome else NEWS_CONTINUE_NUM_PREDICT
                        ),
                    )
                else:
                    local_limit = (
                        SIMPLE_CHAT_NUM_PREDICT if simple_chat
                        else LOCAL_CONVERSATION_NUM_PREDICT if ordinary_no_tool_turn
                        else None
                    )
                    with ollama_chat(
                        model, messages, False, tools,
                        num_predict_override=local_limit,
                    ) as r:
                        data = json.loads(r.read())
            except (TimeoutError, OSError, HTTPError, URLError, json.JSONDecodeError) as exc:
                print(
                    f"[thinkless] local response fallback after "
                    f"{(time.monotonic() - started) * 1000:.0f}ms: {type(exc).__name__}",
                    flush=True,
                )
                out = completed_response(model, LOCAL_TIMEOUT_REPLY, 0, 0)
                self._send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json")
                return
            msg = data.get("message") or {}
            text = clean_model_output(msg.get("content") or "")
            out = completed_response(
                model,
                text,
                int(data.get("prompt_eval_count") or 0),
                int(data.get("eval_count") or 0),
                msg.get("tool_calls") or [],
            )
            self._send(200, json.dumps(out).encode(), "application/json")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        rid = "resp_" + uuid.uuid4().hex[:20]
        mid = "msg_" + uuid.uuid4().hex[:20]
        seq = 0
        full = []
        sanitizer = ModelOutputSanitizer()
        in_tok = 0
        out_tok = 0
        tool_calls: list[dict] = []
        local_started = time.monotonic()
        first_delta_at: float | None = None

        def sse(event: str, obj: dict):
            nonlocal seq
            seq += 1
            obj.setdefault("sequence_number", seq)
            chunk = f"event: {event}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n"
            self.wfile.write(chunk.encode())
            self.wfile.flush()

        sse("response.created", {"type": "response.created", "response": {"id": rid, "status": "in_progress"}})
        sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": mid, "type": "message", "role": "assistant", "status": "in_progress"},
            },
        )

        local_done_reason = ""
        raw_parts: list[str] = []

        def emit_delta(piece: str) -> None:
            nonlocal first_delta_at
            if not piece:
                return
            if first_delta_at is None:
                first_delta_at = time.monotonic()
            full.append(piece)
            sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "delta": piece,
                    "content_index": 0,
                    "item_id": mid,
                    "output_index": 0,
                },
            )

        def consume_stream(chat_messages: list[dict], limit: int | None) -> None:
            nonlocal in_tok, out_tok, local_done_reason, first_delta_at
            with ollama_chat(
                model, chat_messages, True, tools,
                num_predict_override=limit,
            ) as response:
                for raw in response:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    in_tok = int(data.get("prompt_eval_count") or in_tok)
                    out_tok = int(data.get("eval_count") or out_tok)
                    if data.get("done"):
                        local_done_reason = str(data.get("done_reason") or "")
                    raw_piece = ((data.get("message") or {}).get("content")) or ""
                    chunk_calls = ((data.get("message") or {}).get("tool_calls")) or []
                    if chunk_calls:
                        if first_delta_at is None:
                            first_delta_at = time.monotonic()
                        tool_calls.extend(chunk_calls)
                    if raw_piece:
                        raw_parts.append(raw_piece)
                    emit_delta(sanitizer.feed(raw_piece))

        try:
            # Welcomes and unattended news must be complete before any token
            # is exposed. A token-limit cutoff previously reached the room as
            # a half sentence. Interactive dialogue keeps the live stream.
            if fast_welcome or proactive_broadcast:
                kind = "welcome" if fast_welcome else "news"
                data, in_tok, out_tok = _finish_buffered_local_chat(
                    model,
                    messages,
                    tools,
                    kind=kind,
                    first_limit=(
                        WELCOME_NUM_PREDICT if fast_welcome else NEWS_NUM_PREDICT
                    ),
                    retry_limit=(
                        WELCOME_RETRY_NUM_PREDICT
                        if fast_welcome
                        else NEWS_RETRY_NUM_PREDICT
                    ),
                    continue_limit=(
                        None if fast_welcome else NEWS_CONTINUE_NUM_PREDICT
                    ),
                )
                msg = data.get("message") or {}
                raw_text = str(msg.get("content") or "")
                local_done_reason = str(data.get("done_reason") or "")
                emit_delta(sanitizer.feed(raw_text))
                tool_calls.extend(msg.get("tool_calls") or [])
            else:
                local_limit = (
                    SIMPLE_CHAT_NUM_PREDICT if simple_chat
                    else LOCAL_CONVERSATION_NUM_PREDICT if ordinary_no_tool_turn
                    else None
                )
                consume_stream(messages, local_limit)
                raw_text = "".join(raw_parts)
                if (not simple_chat) and _should_finish_incomplete(local_done_reason, raw_text, tool_calls):
                    print(
                        "[thinkless] dialogue incomplete "
                        f"(reason={local_done_reason or 'stop'}); continuing",
                        flush=True,
                    )
                    consume_stream(
                        _continuation_messages(messages, raw_text),
                        DIALOGUE_CONTINUE_NUM_PREDICT,
                    )
        except (TimeoutError, OSError, HTTPError, URLError, json.JSONDecodeError) as exc:
            if not full and not tool_calls:
                first_delta_at = time.monotonic()
                full.append(LOCAL_TIMEOUT_REPLY)
                sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta", "delta": LOCAL_TIMEOUT_REPLY,
                        "content_index": 0, "item_id": mid, "output_index": 0,
                    },
                )
            print(
                f"[thinkless] local stream fallback after "
                f"{(time.monotonic() - local_started) * 1000:.0f}ms: {type(exc).__name__}",
                flush=True,
            )

        tail = sanitizer.feed("", final=True)
        if tail:
            full.append(tail)
            sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "delta": tail,
                    "content_index": 0,
                    "item_id": mid,
                    "output_index": 0,
                },
            )
        text = "".join(full).strip()
        first_ms = (
            (first_delta_at - local_started) * 1000.0
            if first_delta_at is not None else -1.0
        )
        print(
            f"[thinkless] local latency first={first_ms:.0f}ms "
            f"total={(time.monotonic() - local_started) * 1000:.0f}ms "
            f"tools={len(tool_calls)} chars={len(text)} done_reason={local_done_reason or 'unknown'}",
            flush=True,
        )
        sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": mid,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text}],
                },
            },
        )
        output_index = 1
        for call in tool_calls:
            item = _function_call_item(call)
            sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {**item, "status": "in_progress"},
                },
            )
            sse(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"],
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "output_index": output_index,
                    "arguments": item["arguments"],
                },
            )
            sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": item,
                },
            )
            output_index += 1
        sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": completed_response(model, text, in_tok, out_tok, tool_calls),
            },
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    if PREWARM:
        model = os.environ.get("LLM_NAME", "llama3")
        print(f"warming Ollama model {model} (keep_alive={KEEP_ALIVE})", flush=True)
        try:
            with ollama_chat(model, [{"role": "user", "content": "只回复：好"}], False) as response:
                response.read()
        except Exception as exc:  # The HTTP service must still start for diagnostics/retry.
            print(f"Ollama prewarm failed: {exc!r}", flush=True)
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    primary = f", primary={GROK_BASE_URL}/{GROK_MODEL}" if GROK_ENABLED else ""
    print(f"thinkless proxy {HOST}:{PORT} -> {OLLAMA} (think=false){primary}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
