"""Idle semantic refinement for speech-to-speech chat memory.

The normal compactor remains a millisecond-scale local extractor.  This module
patches the in-process Chat class so an optional second pass can reorganize the
same snapshot only while the room is quiet.  Any new activity cancels the pass;
the refined pair is installed only when the chat generation, activity revision,
and both summary item IDs still match.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import weakref
from urllib.request import Request, urlopen

from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import Content as AssistantContent
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent
from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.compaction_prompt import _render_transcript


LOG = logging.getLogger("speech_to_speech.tiered_memory")
ENABLED = os.environ.get("MEMORY_SEMANTIC_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
IDLE_SECONDS = max(3.0, float(os.environ.get("MEMORY_SEMANTIC_IDLE_SECONDS", "12")))
MAX_SECONDS = max(3.0, float(os.environ.get("MEMORY_SEMANTIC_MAX_SECONDS", "15")))
MAX_CHARS = max(300, int(os.environ.get("LLM_COMPACTION_MAX_CHARS", "900")))
NUM_PREDICT = max(64, int(os.environ.get("MEMORY_SEMANTIC_NUM_PREDICT", "256")))
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("LLM_NAME", "llama3")
NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "4096"))
KEEP_ALIVE = os.environ.get("LLM_KEEP_ALIVE", "-1").strip() or "-1"

_ACTIVE_CHATS: weakref.WeakSet[Chat] = weakref.WeakSet()
_ACTIVE_LOCK = threading.Lock()
_SEMANTIC_SLOT = threading.Lock()
_INSTALLED = False


def _keep_alive_value() -> int | str:
    try:
        return int(KEEP_ALIVE)
    except ValueError:
        return KEEP_ALIVE


def _extract_json(text: str) -> dict[str, str]:
    text = re.sub(r"<(?:think|analysis|reasoning)>.*?</(?:think|analysis|reasoning)>", "", text, flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("semantic memory response contained no JSON object")
    data = json.loads(text[start : end + 1])
    user = str(data.get("user_summary", "")).strip()[:MAX_CHARS]
    assistant = str(data.get("assistant_summary", "")).strip()[:MAX_CHARS]
    if not user or not assistant:
        raise ValueError("semantic memory response missed a summary field")
    if not user.startswith("【结构化记忆】"):
        user = f"【结构化记忆】｜{user}"
    if not assistant.startswith("【结构化记忆】"):
        assistant = f"【结构化记忆】｜{assistant}"
    return {"user_summary": user, "assistant_summary": assistant}


def _semantic_request(transcript: str, local_user: str, local_assistant: str, cancel: threading.Event) -> dict[str, str] | None:
    system = """你是直播数字人的会话记忆整理器。把原始对话和本地结构化记忆合并成准确、紧凑的长期记忆。
只输出 JSON，且只能有 user_summary 和 assistant_summary 两个字符串字段。
user_summary 必须使用【结构化记忆】｜身份：...｜偏好：...｜不喜欢：...｜重要信息：...｜近期话题：...。
assistant_summary 必须使用【结构化记忆】｜承诺与结论：...｜近期回复：...。没有内容的栏目可以省略。
用户字段优先保留身份、最新偏好和不喜欢、重要事实、当前话题；后说的信息覆盖矛盾的旧信息。
助手字段优先保留已经给出的结论、承诺、未完成事项。过滤寒暄、主动欢迎和系统指令，不得虚构。"""
    user = f"""本地结构化记忆：
user_summary: {local_user}
assistant_summary: {local_assistant}

原始对话：
{transcript[:8000]}
"""
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": True,
            "think": False,
            "format": "json",
            "keep_alive": _keep_alive_value(),
            "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        },
        ensure_ascii=False,
    ).encode()
    request = Request(f"{OLLAMA}/api/chat", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    pieces: list[str] = []
    started = time.monotonic()
    with urlopen(request, timeout=MAX_SECONDS) as response:
        for raw in response:
            if cancel.is_set() or time.monotonic() - started > MAX_SECONDS:
                response.close()
                return None
            event = json.loads(raw)
            pieces.append(str((event.get("message") or {}).get("content") or ""))
    if cancel.is_set():
        return None
    return _extract_json("".join(pieces))


def _cancel_chat(chat: Chat) -> None:
    event = getattr(chat, "_semantic_memory_cancel", None)
    if event is not None:
        event.set()
    lock = getattr(chat, "_lock", None)
    if lock is not None:
        with lock:
            chat._semantic_memory_revision = getattr(chat, "_semantic_memory_revision", 0) + 1


def cancel_semantic_refinements() -> None:
    """Cancel semantic work as soon as server VAD reports user speech."""
    with _ACTIVE_LOCK:
        chats = list(_ACTIVE_CHATS)
    for chat in chats:
        _cancel_chat(chat)


def _acquire_slot(cancel: threading.Event) -> bool:
    while not cancel.is_set():
        if _SEMANTIC_SLOT.acquire(timeout=0.1):
            return True
    return False


def _schedule_refinement(
    chat: Chat,
    snapshot: list,
    gen: int,
    user_id: str,
    assistant_id: str,
    local_user: str,
    local_assistant: str,
) -> None:
    old = getattr(chat, "_semantic_memory_cancel", None)
    if old is not None:
        old.set()
    cancel = threading.Event()
    chat._semantic_memory_cancel = cancel
    revision = getattr(chat, "_semantic_memory_revision", 0)

    def worker() -> None:
        if cancel.wait(IDLE_SECONDS):
            return
        if not _acquire_slot(cancel):
            return
        started = time.monotonic()
        try:
            if cancel.is_set():
                return
            LOG.info("Idle semantic memory refinement started")
            result = _semantic_request(
                _render_transcript(snapshot), local_user, local_assistant, cancel
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Idle semantic memory refinement failed: %s", exc)
            return
        finally:
            _SEMANTIC_SLOT.release()

        if result is None or cancel.is_set():
            LOG.info("Idle semantic memory refinement cancelled")
            return
        with chat._lock:
            if (
                chat._shutdown.is_set()
                or chat._gen_counter != gen
                or getattr(chat, "_semantic_memory_revision", -1) != revision
                or cancel.is_set()
            ):
                return
            user_item = next((item for item in chat.buffer if item.id == user_id), None)
            assistant_item = next((item for item in chat.buffer if item.id == assistant_id), None)
            if not isinstance(user_item, RealtimeConversationItemUserMessage) or not isinstance(
                assistant_item, RealtimeConversationItemAssistantMessage
            ):
                return
            user_item.content = [UserContent(type="input_text", text=result["user_summary"])]
            assistant_item.content = [AssistantContent(type="output_text", text=result["assistant_summary"])]
        LOG.info("Idle semantic memory refinement applied atomically in %.3fs", time.monotonic() - started)

    thread = threading.Thread(target=worker, daemon=True, name="semantic-memory")
    chat._semantic_memory_thread = thread
    thread.start()
    LOG.info("Semantic memory refinement scheduled after %.1fs idle", IDLE_SECONDS)


def install_tiered_memory() -> None:
    global _INSTALLED
    if _INSTALLED or not ENABLED:
        return
    _INSTALLED = True
    original_init = Chat.__init__
    original_add_item = Chat.add_item
    original_append_tool_output = Chat.append_tool_output
    original_replace = Chat.replace_user_message_text
    original_remove = Chat.remove_user_message
    original_reset = Chat.reset
    original_close = Chat.close
    original_compact_worker = Chat._compact_worker

    def patched_init(self, size):
        original_init(self, size)
        self._semantic_memory_cancel = threading.Event()
        self._semantic_memory_revision = 0
        self._semantic_memory_thread = None
        with _ACTIVE_LOCK:
            _ACTIVE_CHATS.add(self)

    def touch(self):
        _cancel_chat(self)

    def patched_add_item(self, item):
        touch(self)
        return original_add_item(self, item)

    def patched_append_tool_output(self, call_id, output_item):
        touch(self)
        return original_append_tool_output(self, call_id, output_item)

    def patched_replace(self, item_id, text):
        touch(self)
        return original_replace(self, item_id, text)

    def patched_remove(self, item_id):
        touch(self)
        return original_remove(self, item_id)

    def patched_reset(self):
        touch(self)
        return original_reset(self)

    def patched_close(self):
        touch(self)
        return original_close(self)

    def patched_compact_worker(self, compactor, snapshot, marker_ids, gen):
        original_compact_worker(self, compactor, snapshot, marker_ids, gen)
        if self._shutdown.is_set() or self._gen_counter != gen:
            return
        with self._lock:
            if len(self.buffer) < 2:
                return
            user_item, assistant_item = self.buffer[0], self.buffer[1]
            if not isinstance(user_item, RealtimeConversationItemUserMessage) or not isinstance(
                assistant_item, RealtimeConversationItemAssistantMessage
            ):
                return
            local_user = " ".join(part.text or "" for part in user_item.content if part.type == "input_text")
            local_assistant = " ".join(
                part.text or "" for part in assistant_item.content if part.type == "output_text"
            )
            user_id, assistant_id = user_item.id, assistant_item.id
        if user_id and assistant_id and local_user and local_assistant:
            _schedule_refinement(
                self, snapshot, gen, user_id, assistant_id, local_user, local_assistant
            )

    Chat.__init__ = patched_init
    Chat.add_item = patched_add_item
    Chat.append_tool_output = patched_append_tool_output
    Chat.replace_user_message_text = patched_replace
    Chat.remove_user_message = patched_remove
    Chat.reset = patched_reset
    Chat.close = patched_close
    Chat._compact_worker = patched_compact_worker
    LOG.info(
        "Tiered memory enabled: local structured compaction + cancellable semantic refinement after %.1fs idle",
        IDLE_SECONDS,
    )
