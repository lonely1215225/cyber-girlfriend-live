#!/usr/bin/env python3
"""OpenAI /v1/responses shim in front of Ollama /api/chat with think=false.

Ollama's /v1/responses ignores think=false and spends 20s+ on hidden reasoning.
/api/chat?think=false replies in ~1s. speech-to-speech talks to this shim.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
LOCAL_TIMEOUT_REPLY = "嗯？刚才没接稳，你再说一遍嘛。"
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
_EXTERNAL_INTENT_RE = re.compile(
    r"(?:查(?:一下|下|查)?|搜索|搜一下|联网|核实|验证|最新|实时|新闻|热搜|行情|报价|汇率|"
    r"现在.{0,12}(?:价格|多少钱|报价|行情|天气|汇率)|"
    r"今天.{0,10}(?:新闻|天气|价格|发生)|"
    r"最近.{0,14}(?:上涨|下跌|涨|跌|原因)|"
    r"为什么.{0,14}(?:上涨|下跌|涨|跌)|"
    r"\b(?:current|latest|today'?s?)\s+(?:price|news|weather|market)\b)",
    re.IGNORECASE,
)


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
        "直播间入场欢迎生成器" in str(message.get("content", ""))
        for message in messages
    )


def _is_proactive_broadcast_request(messages: list[dict]) -> bool:
    """Recognize server-owned unattended broadcasts, never user semantics."""
    return any(
        "无人连线时的直播间主动播报" in str(message.get("content", ""))
        for message in messages
    )


def _is_fast_discovery_turn(payload: dict) -> bool:
    """Use the resident model for native bounded tool selection.

    Ordinary conversation is answered in this same turn. If external evidence
    is genuinely needed, the model calls the single progressive-disclosure
    tool. Once a capability result exists, research returns to the stronger
    remote model.
    """
    tools = [tool for tool in payload.get("tools") or [] if isinstance(tool, dict)]
    names = {str(tool.get("name") or "") for tool in tools}
    if names != {"request_external_capabilities"}:
        return False
    return not any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in payload.get("input") or []
    )


def _needs_reliable_external_route(messages: list[dict]) -> bool:
    """Recognize explicit freshness/lookup requests that must not be hallucinated.

    This is a routing safety net, not an answer generator. The model still
    chooses the concrete capability and tools, then writes from their evidence.
    Ordinary conversation never pays this second planning hop.
    """
    current = next(
        (
            str(item.get("content") or "")
            for item in reversed(messages)
            if item.get("role") == "user"
        ),
        "",
    )
    current = current.split("\n\n【", 1)[0].strip()
    return bool(_EXTERNAL_INTENT_RE.search(current))


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
    num_predict = int(os.environ.get("LLM_NUM_PREDICT", "128"))
    if _is_compaction_request(messages):
        return COMPACTION_NUM_PREDICT
    if _is_room_welcome_request(messages):
        return WELCOME_NUM_PREDICT
    return num_predict


def ollama_chat(
    model: str,
    messages: list[dict],
    stream: bool,
    tools: list[dict] | None = None,
    *,
    num_predict_override: int | None = None,
):
    # Default Ollama ctx for this 9B is 262144. Prompt eval then takes ~8s even
    # for a short line, and under GPU contention it exceeds the s2s 20s timeout.
    num_ctx = int(os.environ.get("LLM_NUM_CTX", "4096"))
    num_predict = (
        max(1, int(num_predict_override))
        if num_predict_override is not None
        else _num_predict_for_messages(messages)
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
    return urlopen(req, timeout=LOCAL_READ_TIMEOUT_SECONDS)


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
                "根据对方刚说的话接住语气或情绪，甜甜的、灵动、有点坏但不刻薄。"
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

    if eager_prefix and prefix_text:
        write_event(
            "response.created",
            {
                "type": "response.created",
                "response": {"id": synthetic_response_id, "status": "in_progress"},
            },
        )
        inject_prefix()

    def emit(lines: list[str]) -> None:
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


def _function_call_item(call: dict) -> dict:
    function = call.get("function") or {}
    arguments = function.get("arguments") or {}
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": "fc_" + uuid.uuid4().hex[:20],
        "type": "function_call",
        "call_id": call.get("id") or "call_" + uuid.uuid4().hex[:20],
        "name": function.get("name") or "",
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
        explicit_external_request = fast_discovery and _needs_reliable_external_route(messages)
        has_tool_evidence = any(
            isinstance(item, dict) and item.get("type") == "function_call_output"
            for item in req.get("input") or []
        )
        # One request must have exactly one prose generator.  The previous
        # hybrid path exposed a local answer and then appended a second Grok
        # answer, which sounded like repeated replies and delayed newer room
        # comments.  The resident model now owns bounded low-latency turns;
        # Grok owns evidence-backed synthesis after real tool output.
        local_primary = (
            fast_discovery
            or fast_conversation
            or fast_external_planning
            or fast_welcome
        )
        if explicit_external_request:
            current = next(
                (str(item.get("content") or "") for item in reversed(messages) if item.get("role") == "user"),
                "",
            ).split("\n\n【", 1)[0].strip()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "External routing only. Call request_external_capabilities exactly once with "
                        "the smallest required capability set. Do not answer, estimate, mention a tool, "
                        "or claim a result before verified tool output."
                    ),
                },
                {"role": "user", "content": current},
            ]
        elif fast_discovery:
            # Tool schemas are intentionally absent for ordinary conversation.
            # This makes the no-tool route an invariant instead of trusting a
            # small model to obey a descriptive hint while a tool is present.
            # The model still writes the answer; this guard only decides
            # whether external I/O is permitted for the current user request.
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

        if COMPACTION_MODE == "local" and _is_compaction_request(messages):
            text = local_compaction(messages)
            out = completed_response(model, text, 0, 0)
            self._send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json")
            return

        # Grok owns every generative/semantic task. Local Ollama remains only
        # a provider-outage fallback plus deterministic compaction/exact TTS.
        # Tool permission and public-output safety are enforced below in code,
        # independently of either model's instruction following.
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
                grok_request = req
                if use_local_lead:
                    grok_request = dict(req)
                    grok_request["instructions"] = (
                        str(req.get("instructions") or "")
                        + "\n同一轮会先播放一句本地生成的简短接话，它已经负责接住情绪、"
                        "态度和第一层直接回应。请从补充信息、解释或推进话题开始，"
                        "不要再次问候、安慰、复述问题或重复第一层结论。"
                    ).strip()
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
                with ollama_chat(model, messages, False, tools) as r:
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
        try:
            # A welcome is short, but it must be delivered atomically. Buffer
            # it until Ollama reports a clean finish so a token-limit response
            # can never become a half-spoken greeting. Normal dialogue keeps
            # the existing token stream and therefore its minimum TTFT.
            if fast_welcome:
                with ollama_chat(model, messages, False, tools) as r:
                    data = json.loads(r.read())
                msg = data.get("message") or {}
                raw_text = str(msg.get("content") or "")
                local_done_reason = str(data.get("done_reason") or "")
                in_tok = int(data.get("prompt_eval_count") or 0)
                out_tok = int(data.get("eval_count") or 0)
                if local_done_reason == "length":
                    print(
                        "[thinkless] welcome hit output limit; regenerating a shorter complete line",
                        flush=True,
                    )
                    retry_messages = [
                        *messages,
                        {"role": "assistant", "content": raw_text},
                        {
                            "role": "user",
                            "content": (
                                "上一句因长度限制没有说完。请重新生成一句更短、语义完整的欢迎词；"
                                "仍遵守原要求，并用句号、问号、感叹号或波浪号自然收尾。"
                            ),
                        },
                    ]
                    with ollama_chat(
                        model,
                        retry_messages,
                        False,
                        tools,
                        num_predict_override=WELCOME_RETRY_NUM_PREDICT,
                    ) as r:
                        data = json.loads(r.read())
                    msg = data.get("message") or {}
                    raw_text = str(msg.get("content") or "")
                    local_done_reason = str(data.get("done_reason") or "")
                    in_tok += int(data.get("prompt_eval_count") or 0)
                    out_tok += int(data.get("eval_count") or 0)
                piece = sanitizer.feed(raw_text)
                if piece:
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
                tool_calls.extend(msg.get("tool_calls") or [])
            else:
                with ollama_chat(model, messages, True, tools) as r:
                    for raw in r:
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
                        piece = sanitizer.feed(raw_piece)
                        if piece:
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
