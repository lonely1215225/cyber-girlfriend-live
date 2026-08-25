import sys
import asyncio
import json
import time
import unittest
from unittest import mock
from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "s2s" / "hf-realtime-voice"
sys.path.insert(0, str(FRONTEND_DIR))

from mention_reply import (  # noqa: E402
    MentionReplyWorker,
    looks_like_deferred_answer,
    parse_mention,
)


class MentionReplyTests(unittest.TestCase):
    def test_extracts_prompt_after_mention(self):
        request = parse_mention({"id": "m1", "speaker": "林清欢", "text": "@小麻，你喜欢猫吗"})
        self.assertIsNotNone(request)
        self.assertEqual(request.prompt, "你喜欢猫吗")

    def test_bare_mention_gets_natural_fallback(self):
        request = parse_mention({"id": "m2", "speaker": "Avery Blake", "text": "@小麻"})
        self.assertIn("自然地回应", request.prompt)

    def test_normal_chat_does_not_trigger(self):
        self.assertIsNone(parse_mention({"id": "m3", "speaker": "观众", "text": "大家好"}))

    def test_detects_transition_promise_instead_of_final_answer(self):
        self.assertTrue(looks_like_deferred_answer("好哒，我去查查最新新闻"))
        self.assertFalse(looks_like_deferred_answer("比特币今天上涨，主要与资金流入有关"))

class FakeRoom:
    async def can_bot_reply(self):
        return True


class MentionResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_one_proactive_room_topic_can_be_pending(self):
        worker = MentionReplyWorker(FakeRoom(), mock.Mock(), "ws://unused")
        self.assertTrue(worker.enqueue_proactive("讲一条新闻"))
        self.assertFalse(worker.enqueue_proactive("再讲一条新闻"))
        self.assertEqual(len(worker.pending), 1)
        self.assertTrue(worker.pending[0].proactive)

    async def test_model_native_tool_calls_run_in_parallel_with_progress(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item
            async def publish_agent_job(self, job, **_kwargs): self.items.append(dict(job)); return job
            async def participant_memory_context(self, *_args): return ""
            async def active_news_context(self, *_args): return ""

        class ToolGateway:
            enabled = True
            DISCOVERY_TOOL_NAME = "request_external_capabilities"
            def __init__(self): self.calls = []
            def discovery_tool(self):
                return {
                    "type": "function", "name": self.DISCOVERY_TOOL_NAME,
                    "description": "按需申请外部能力",
                    "parameters": {"type": "object", "properties": {
                        "capabilities": {"type": "array", "items": {"type": "string"}},
                    }},
                }
            async def list_tools(self):
                return [{
                    "type": "function", "name": name, "description": "查询资料",
                    "parameters": {"type": "object", "properties": {}},
                    "progress_text": "我正在核对两个独立来源，稍等一下呀。",
                } for name in ("source_one", "source_two")]
            async def tools_for_capabilities(self, capabilities):
                self.requested = capabilities
                return await self.list_tools()
            async def call(self, name, arguments):
                self.calls.append((name, arguments, time.monotonic()))
                await asyncio.sleep(0.05)
                return json.dumps({"source": name, "result": "可信资料"}, ensure_ascii=False)

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
                    '{"type":"response.function_call_arguments.done","name":"request_external_capabilities","arguments":"{\\"capabilities\\":[\\"web\\"]}","call_id":"d1"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                    '{"type":"response.function_call_arguments.done","name":"source_one","arguments":"{}","call_id":"c1"}',
                    '{"type":"response.function_call_arguments.done","name":"source_two","arguments":"{}","call_id":"c2"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                    '{"type":"response.audio_transcript.done","transcript":"这是核对后的最终答案。"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                ])
                self.sent = []
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def recv(self): return next(self.events)
            async def send(self, message): self.sent.append(json.loads(message))

        room, gateway, socket = RecordingRoom(), ToolGateway(), FakeWebSocket()
        worker = MentionReplyWorker(room, gateway, "ws://unused")
        request = parse_mention({"id": "tools", "participant_id": "p1", "speaker": "观众", "text": "@小麻 查证一下"})
        worker._speak_exact = mock.AsyncMock()
        started = time.monotonic()
        with mock.patch("mention_reply.websockets.connect", return_value=socket):
            await worker._respond(request)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.09)
        self.assertEqual({name for name, _, _ in gateway.calls}, {"source_one", "source_two"})
        worker._speak_exact.assert_awaited_once()
        self.assertTrue(any(item.get("partial") and "同时查" in item["text"] for item in room.items))
        self.assertTrue(any(item.get("text") == "这是核对后的最终答案。" for item in room.items))

    async def test_proactive_speech_streams_to_room_and_is_finalized(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item

        class ToolsMustNotBeLoaded:
            enabled = True
            async def list_tools(self):
                raise AssertionError("proactive speech must use its preloaded evidence")

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
                    '{"type":"response.audio_transcript.delta","delta":"第一句新闻。"}',
                    '{"type":"response.audio_transcript.done","transcript":"第一句新闻。"}',
                    '{"type":"response.audio_transcript.done","transcript":"第二句也要显示。"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                ])
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def recv(self): return next(self.events)
            async def send(self, _message): pass

        room = RecordingRoom()
        worker = MentionReplyWorker(room, ToolsMustNotBeLoaded(), "ws://unused")
        worker.enqueue_proactive("讲新闻")
        request = worker.pending.popleft()
        with mock.patch("mention_reply.websockets.connect", return_value=FakeWebSocket()):
            await worker._respond(request)
        self.assertTrue(any(item.get("partial") for item in room.items))
        final = [item for item in room.items if not item.get("partial")][-1]
        self.assertEqual(final["text"], "第一句新闻。第二句也要显示。")
        self.assertFalse(final.get("interrupted", False))

    async def test_spoken_prefix_survives_connection_failure(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item

        class NoTools:
            enabled = False

        class FailingWebSocket:
            def __init__(self): self.index = 0
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def send(self, _message): pass
            async def recv(self):
                self.index += 1
                if self.index == 1: return '{"type":"session.created"}'
                if self.index == 2: return '{"type":"response.audio_transcript.delta","delta":"已经播出的半句话"}'
                raise ConnectionError("stream lost")

        room = RecordingRoom()
        worker = MentionReplyWorker(room, NoTools(), "ws://unused")
        worker.enqueue_proactive("讲新闻")
        with mock.patch("mention_reply.websockets.connect", return_value=FailingWebSocket()):
            with self.assertRaises(ConnectionError):
                await worker._respond(worker.pending.popleft())
        final = [item for item in room.items if not item.get("partial")][-1]
        self.assertEqual(final["text"], "已经播出的半句话")
        self.assertTrue(final["interrupted"])


if __name__ == "__main__":
    unittest.main()
