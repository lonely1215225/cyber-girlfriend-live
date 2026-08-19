#!/usr/bin/env python3
"""OpenAI /v1/responses shim in front of Ollama /api/chat with think=false.

Ollama's /v1/responses ignores think=false and spends 20s+ on hidden reasoning.
/api/chat?think=false replies in ~1s. speech-to-speech talks to this shim.
"""
from __future__ import annotations

import json
import os
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
HOST = os.environ.get("THINKLESS_HOST", "127.0.0.1")
PORT = int(os.environ.get("THINKLESS_PORT", "11435"))


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
        role = item.get("role") or "user"
        if role not in ("system", "user", "assistant"):
            role = "user"
        text = _extract_text(item.get("content"))
        if text.strip():
            msgs.append({"role": role, "content": text})
    return msgs or [{"role": "user", "content": "你好"}]


def ollama_chat(model: str, messages: list[dict], stream: bool):
    # Default Ollama ctx for this 9B is 262144. Prompt eval then takes ~8s even
    # for a short line, and under GPU contention it exceeds the s2s 20s timeout.
    num_ctx = int(os.environ.get("LLM_NUM_CTX", "4096"))
    num_predict = int(os.environ.get("LLM_NUM_PREDICT", "128"))
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": False,
            "options": {"num_ctx": num_ctx, "num_predict": num_predict},
        }
    ).encode()
    req = Request(
        f"{OLLAMA}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urlopen(req, timeout=120)


def completed_response(model: str, text: str, in_tok: int, out_tok: int) -> dict:
    rid = "resp_" + uuid.uuid4().hex[:20]
    mid = "msg_" + uuid.uuid4().hex[:20]
    return {
        "id": rid,
        "object": "response",
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": mid,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
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
        want_stream = bool(req.get("stream"))

        if not want_stream:
            with ollama_chat(model, messages, False) as r:
                data = json.loads(r.read())
            msg = data.get("message") or {}
            text = msg.get("content") or ""
            out = completed_response(
                model,
                text,
                int(data.get("prompt_eval_count") or 0),
                int(data.get("eval_count") or 0),
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
        in_tok = 0
        out_tok = 0

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

        with ollama_chat(model, messages, True) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                data = json.loads(line)
                in_tok = int(data.get("prompt_eval_count") or in_tok)
                out_tok = int(data.get("eval_count") or out_tok)
                piece = ((data.get("message") or {}).get("content")) or ""
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

        text = "".join(full)
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
        sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": completed_response(model, text, in_tok, out_tok),
            },
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"thinkless proxy {HOST}:{PORT} -> {OLLAMA} (think=false)", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
