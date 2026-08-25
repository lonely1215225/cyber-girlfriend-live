#!/usr/bin/env python3
"""OpenAI /v1/responses shim in front of Ollama /api/chat with think=false.

Ollama's /v1/responses ignores think=false and spends 20s+ on hidden reasoning.
/api/chat?think=false replies in ~1s. speech-to-speech talks to this shim.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from memory_compaction import local_compaction as _local_compaction

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
HOST = os.environ.get("THINKLESS_HOST", "127.0.0.1")
PORT = int(os.environ.get("THINKLESS_PORT", "11435"))
KEEP_ALIVE = os.environ.get("LLM_KEEP_ALIVE", "-1").strip() or "-1"
PREWARM = os.environ.get("LLM_PREWARM", "1").strip().lower() in {"1", "true", "yes", "on"}
COMPACTION_NUM_PREDICT = int(os.environ.get("LLM_COMPACTION_NUM_PREDICT", "256"))
COMPACTION_MODE = os.environ.get("LLM_COMPACTION_MODE", "local").strip().lower()
COMPACTION_MAX_CHARS = max(300, int(os.environ.get("LLM_COMPACTION_MAX_CHARS", "900")))
GROK_ENABLED = os.environ.get("GROK_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
GROK_BASE_URL = os.environ.get("GROK_PROXY_BASE_URL", "http://127.0.0.1:18080/v1").rstrip("/")
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-4.6").strip() or "grok-4.6"
GROK_API_KEY = os.environ.get("GROK_PROXY_API_KEY", "").strip()
GROK_REASONING_EFFORT = os.environ.get("GROK_REASONING_EFFORT", "low").strip().lower() or "low"
GROK_TIMEOUT_SECONDS = max(2.0, float(os.environ.get("GROK_TIMEOUT_SECONDS", "45")))
_REASONING_TAGS = {"think", "analysis", "reasoning"}


class ModelOutputSanitizer:
    """Incrementally remove provider reasoning blocks without delaying speech.

    Model tokens and XML-like tags can be split across arbitrary NDJSON chunks,
    so a regex on each individual chunk is not sufficient.
    """

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
                    looks_like_reasoning = any(tag.startswith(candidate) for tag in _REASONING_TAGS)
                    if not self.suppressed_depth and not looks_like_reasoning:
                        output.append(self.buffer)
                    self.buffer = ""
                break
            raw_tag = self.buffer[: closing + 1]
            tag_body = self.buffer[1:closing].strip()
            is_closing = tag_body.startswith("/")
            tag_name = re.split(r"\s+", tag_body.lstrip("/").rstrip("/"), maxsplit=1)[0].lower()
            if tag_name in _REASONING_TAGS:
                if is_closing:
                    self.suppressed_depth = max(0, self.suppressed_depth - 1)
                elif not tag_body.endswith("/"):
                    self.suppressed_depth += 1
            elif not self.suppressed_depth:
                # Preserve unknown angle-bracket text; only provider reasoning
                # markup is hidden.
                output.append(raw_tag)
            self.buffer = self.buffer[closing + 1 :]
        return "".join(output)


def clean_model_output(text: str) -> str:
    cleaner = ModelOutputSanitizer()
    return cleaner.feed(text, final=True).strip()


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
    """Arrival greetings are short creative work suited to the resident LLM."""
    return any(
        "直播间入场欢迎生成器" in str(message.get("content", ""))
        for message in messages
    )


def _is_fast_discovery_turn(payload: dict) -> bool:
    """Use the resident model for the lightweight first turn.

    The first turn can either answer ordinary conversation immediately or ask
    for external capabilities.  Once a capability result exists, subsequent
    research and synthesis return to the stronger remote model.
    """
    tools = [tool for tool in payload.get("tools") or [] if isinstance(tool, dict)]
    names = {str(tool.get("name") or "") for tool in tools}
    if names != {"request_external_capabilities"}:
        return False
    return not any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in payload.get("input") or []
    )


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


def ollama_chat(model: str, messages: list[dict], stream: bool, tools: list[dict] | None = None):
    # Default Ollama ctx for this 9B is 262144. Prompt eval then takes ~8s even
    # for a short line, and under GPU contention it exceeds the s2s 20s timeout.
    num_ctx = int(os.environ.get("LLM_NUM_CTX", "4096"))
    num_predict = int(os.environ.get("LLM_NUM_PREDICT", "128"))
    if _is_compaction_request(messages):
        num_predict = COMPACTION_NUM_PREDICT
    elif _is_room_welcome_request(messages):
        num_predict = min(num_predict, 48)
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
    return urlopen(req, timeout=120)


def grok_response(payload: dict):
    """Forward one Responses API request to the private Grok OAuth proxy.

    The public application continues to address this shim with its logical
    model name.  Provider-specific model rewriting happens only here, so the
    same request can safely fall back to the local Ollama model.
    """
    forwarded = dict(payload)
    forwarded["model"] = GROK_MODEL
    forwarded.setdefault("reasoning", {"effort": GROK_REASONING_EFFORT})
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
        messages = to_messages(req.get("input"))
        tools = to_ollama_tools(req.get("tools"))
        want_stream = bool(req.get("stream"))
        fast_discovery = _is_fast_discovery_turn(req)
        fast_conversation = _is_fast_conversation_followup(req)
        fast_external_planning = _is_fast_external_planning(req)
        fast_welcome = _is_room_welcome_request(messages)
        if fast_discovery:
            # Routing must not see old prices/news from memory: that made an
            # ordinary "I'm back" continuation request web access.  The full
            # history remains in the original request for the answer turn.
            current = next(
                (str(item.get("content") or "") for item in reversed(messages) if item.get("role") == "user"),
                "",
            )
            current = current.split("\n\n【", 1)[0].strip()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Route only; do not answer. Call request_external_capabilities once. "
                        "Use conversation for chat, roleplay, opinions or continuation. Use the smallest "
                        "external set only for current facts, news, prices, lookup or verification."
                    ),
                },
                {"role": "user", "content": current},
            ]
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

        # Grok is the high-quality primary provider.  Connection, DNS, OAuth,
        # quota, and 5xx failures fall back before any response headers are
        # emitted, preserving the existing local low-latency path.
        if (
            GROK_ENABLED
            and not _is_exact_speech_request(messages)
            and not fast_discovery
            and not fast_conversation
            and not fast_external_planning
            and not fast_welcome
        ):
            try:
                upstream = grok_response(req)
                if not want_stream:
                    with upstream:
                        data = json.loads(upstream.read())
                    data["output_text"] = response_output_text(data)
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
                    while True:
                        chunk = upstream.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                return
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
                print(f"[thinkless] Grok unavailable, using local fallback: {type(exc).__name__}: {exc}", flush=True)

        if not want_stream:
            with ollama_chat(model, messages, False, tools) as r:
                data = json.loads(r.read())
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

        with ollama_chat(model, messages, True, tools) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                data = json.loads(line)
                in_tok = int(data.get("prompt_eval_count") or in_tok)
                out_tok = int(data.get("eval_count") or out_tok)
                raw_piece = ((data.get("message") or {}).get("content")) or ""
                chunk_calls = ((data.get("message") or {}).get("tool_calls")) or []
                if chunk_calls:
                    tool_calls.extend(chunk_calls)
                piece = sanitizer.feed(raw_piece)
                if piece:
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
