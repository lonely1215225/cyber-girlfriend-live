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

    async def publish_agent_job(self, job, **_kwargs):
        return job


class MentionResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_mention_publishes_queue_status_while_call_is_busy(self):
        class BusyRoom:
            def __init__(self): self.jobs = []
            async def can_bot_reply(self): return False
            async def publish_agent_job(self, job, **kwargs):
                self.jobs.append((dict(job), kwargs)); return job

        room = BusyRoom()
        worker = MentionReplyWorker(room, mock.Mock(), "ws://unused")
        message = {
            "id": "m-queued", "participant_id": "p1", "speaker": "林清欢",
            "text": "@小麻 你好",
        }
        position = worker.enqueue(message)
        await worker.publish_queued(message, position)
        job, kwargs = room.jobs[-1]
        self.assertEqual(job["phase"], "queued")
        self.assertIn("第 1 位", job["status_text"])
        self.assertFalse(job["terminal"])
        self.assertEqual(kwargs["reply_to"]["speaker"], "林清欢")

    async def test_restart_recovers_recent_mention_without_agent_job(self):
        now = time.time()
        class FakeStore:
            async def load_agent_jobs(self, **kwargs): return []
            async def load_recent_messages(self, _limit):
                return [
                    {"id": "plain", "participant_id": "p1", "speaker": "林清欢",
                     "text": "普通聊天", "created_at": now},
                    {"id": "mention", "participant_id": "p1", "speaker": "林清欢",
                     "text": "@小麻 现在比特币多少钱", "created_at": now},
                ]
        class RecoverRoom:
            store = FakeStore()
            def __init__(self): self.jobs = []
            async def can_bot_reply(self): return False
            async def publish_agent_job(self, job, **_kwargs): self.jobs.append(dict(job)); return job

        room = RecoverRoom()
        worker = MentionReplyWorker(room, mock.Mock(), "ws://unused")
        await worker.restore_jobs()
        self.assertEqual([item.message_id for item in worker.pending], ["mention"])
        self.assertEqual(room.jobs[-1]["phase"], "queued")

    async def test_welcome_queue_is_deduplicated_and_prioritized(self):
        worker = MentionReplyWorker(FakeRoom(), mock.Mock(), "ws://unused")
        worker.enqueue_proactive("讲一条新闻")
        self.assertTrue(worker.enqueue_welcome(participant_id="p1", speaker="林清欢"))
        self.assertFalse(worker.enqueue_welcome(participant_id="p1", speaker="林清欢"))
        worker.enqueue({"id": "m1", "participant_id": "p2", "speaker": "Nova", "text": "@小麻 在吗"})
        self.assertEqual(
            [(item.welcome, item.proactive) for item in worker.pending],
            # A direct question makes an unplayed idle broadcast stale.
            [(False, False), (True, False)],
        )

    async def test_live_call_discards_only_that_callers_stale_welcome(self):
        worker = MentionReplyWorker(FakeRoom(), mock.Mock(), "ws://unused")
        worker.enqueue_welcome(participant_id="p1", speaker="林清欢")
        worker.enqueue_welcome(participant_id="p2", speaker="Nova")
        worker.enqueue({"id": "m1", "participant_id": "p3", "speaker": "Emery", "text": "@小麻 在吗"})
        self.assertEqual(worker.drop_welcomes("p1"), 1)
        self.assertEqual(
            [(item.participant_id, item.welcome) for item in worker.pending],
            [("p3", False), ("p2", True)],
        )

    async def test_same_comment_cannot_be_queued_twice(self):
        worker = MentionReplyWorker(FakeRoom(), mock.Mock(), "ws://unused")
        message = {"id": "m1", "participant_id": "p1", "speaker": "林清欢", "text": "@小麻 在吗"}
        self.assertEqual(worker.enqueue(message), 1)
        self.assertEqual(worker.enqueue(message), 1)
        self.assertEqual([item.message_id for item in worker.pending], ["m1"])
        worker._active_request = worker.pending.popleft()
        self.assertEqual(worker.enqueue(message), 0)
        self.assertEqual(len(worker.pending), 0)

    async def test_new_message_drops_stale_welcome_proactive_and_same_viewer_question(self):
        worker = MentionReplyWorker(FakeRoom(), mock.Mock(), "ws://unused")
        worker.enqueue_welcome(participant_id="p1", speaker="林清欢")
        worker.enqueue_proactive("讲一条新闻")
        worker.enqueue({"id": "old", "participant_id": "p1", "speaker": "林清欢", "text": "@小麻 在吗"})
        worker.enqueue({"id": "new", "participant_id": "p1", "speaker": "林清欢", "text": "@小麻 你是谁"})
        self.assertEqual([item.message_id for item in worker.pending], ["new"])

    async def test_new_message_cancels_unspoken_active_request(self):
        worker = MentionReplyWorker(FakeRoom(), mock.Mock(), "ws://unused")
        old = parse_mention({
            "id": "old", "participant_id": "p1", "speaker": "林清欢", "text": "@小麻 在吗",
        })
        worker._active_request = old
        worker._response_task = asyncio.create_task(asyncio.sleep(30))
        worker.enqueue({"id": "new", "participant_id": "p1", "speaker": "林清欢", "text": "@小麻 你是谁"})
        await asyncio.sleep(0)
        self.assertTrue(old.superseded)
        self.assertTrue(worker._response_task.cancelled())

    async def test_delivered_reply_is_terminal_instead_of_requeued_after_call_preemption(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.jobs = []
            async def publish_agent_job(self, job, **_kwargs):
                self.jobs.append(dict(job)); return job

        room = RecordingRoom()
        worker = MentionReplyWorker(room, mock.Mock(), "ws://unused")
        request = parse_mention({
            "id": "m-delivered", "participant_id": "p1", "speaker": "林清欢",
            "text": "@小麻 在吗",
        })
        request.delivery_started = True
        request.final_delivered = True
        request.delivered_text = "在呀，刚好在等你。"
        await worker._finalize_interrupted_delivery(request)
        self.assertTrue(room.jobs[-1]["terminal"])
        self.assertEqual(room.jobs[-1]["phase"], "completed")
        self.assertEqual(room.jobs[-1]["final_text"], "在呀，刚好在等你。")

        request.final_delivered = False
        request.delivered_text = "在呀"
        await worker._finalize_interrupted_delivery(request)
        self.assertEqual(room.jobs[-1]["phase"], "cancelled")
        self.assertEqual(room.jobs[-1]["error"], "preempted_after_delivery")

    async def test_welcome_uses_persona_without_loading_tools(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item

        class ToolsMustNotBeLoaded:
            enabled = True
            async def list_tools(self):
                raise AssertionError("welcome must not load research tools")

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
                    '{"type":"response.audio_transcript.done","transcript":"林清欢，你一来，今晚的月亮都像偷偷调亮了一格呀。"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                ])
                self.sent = []
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def recv(self): return next(self.events)
            async def send(self, message): self.sent.append(json.loads(message))

        room, socket = RecordingRoom(), FakeWebSocket()
        worker = MentionReplyWorker(
            room, ToolsMustNotBeLoaded(), "ws://unused",
            persona_provider=mock.AsyncMock(return_value="你叫小麻，性格甜美灵动。"),
        )
        worker.enqueue_welcome(participant_id="p1", speaker="林清欢")
        with mock.patch("mention_reply.websockets.connect", return_value=socket):
            await worker._respond(worker.pending.popleft())
        session = next(item["session"] for item in socket.sent if item.get("type") == "session.update")
        user_item = next(item for item in socket.sent if item.get("type") == "conversation.item.create")
        self.assertIn("甜美灵动", session["instructions"])
        self.assertIn("直播间入场欢迎生成器", session["instructions"])
        self.assertIn("林清欢", user_item["item"]["content"][0]["text"])
        self.assertEqual(room.items[-1]["text"], "林清欢，你一来，今晚的月亮都像偷偷调亮了一格呀。")
        self.assertIsNone(room.items[-1]["reply_to"])
        self.assertEqual(room.items[-1]["memory_user_id"], "p1")

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
        session_updates = [item["session"] for item in socket.sent if item.get("type") == "session.update"]
        self.assertEqual(session_updates[0].get("tool_choice"), "auto")
        self.assertNotIn("conversation", session_updates[0]["instructions"])
        self.assertTrue(any(item.get("tool_choice") == "required" and len(item.get("tools") or []) == 2
                            for item in session_updates))
        self.assertTrue(any(item.get("tool_choice") == "auto" for item in session_updates))

    async def test_external_route_cannot_complete_before_a_real_tool_call(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.jobs = []
            async def publish_bot_reply(self, **item): return item
            async def publish_agent_job(self, job, **_kwargs): self.jobs.append(dict(job)); return job
            async def participant_memory_context(self, *_args): return ""
            async def active_news_context(self, *_args): return ""

        class ToolGateway:
            enabled = True
            DISCOVERY_TOOL_NAME = "request_external_capabilities"
            def discovery_tool(self):
                return {
                    "type": "function", "name": self.DISCOVERY_TOOL_NAME,
                    "description": "按需申请外部能力", "parameters": {"type": "object"},
                }
            async def list_tools(self):
                return [{
                    "type": "function", "name": "smart_web_search", "description": "联网查询",
                    "parameters": {"type": "object"}, "progress_text": "我先帮你查清楚呀。",
                }]
            async def tools_for_capabilities(self, _capabilities): return await self.list_tools()
            async def call(self, _name, _arguments): return '{"result":"小米汽车可信资料"}'

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
                    '{"type":"response.function_call_arguments.done","name":"request_external_capabilities","arguments":"{\\"capabilities\\":[\\"web\\"]}","call_id":"d1"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                    # Simulate a provider ignoring tool_choice and returning only a promise.
                    '{"type":"response.audio_transcript.done","transcript":"马上回来告诉你。"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                    '{"type":"response.function_call_arguments.done","name":"smart_web_search","arguments":"{\\"query\\":\\"小米最新汽车价格\\"}","call_id":"c1"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                    '{"type":"response.audio_transcript.done","transcript":"这是查询后的具体车型和价格。"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                ])
                self.sent = []
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def recv(self): return next(self.events)
            async def send(self, message): self.sent.append(json.loads(message))

        room, socket = RecordingRoom(), FakeWebSocket()
        worker = MentionReplyWorker(room, ToolGateway(), "ws://unused")
        worker._speak_exact = mock.AsyncMock()
        request = parse_mention({
            "id": "car", "participant_id": "p1", "speaker": "观众",
            "text": "@小麻 小米最新的汽车是什么，多少钱",
        })
        with mock.patch("mention_reply.websockets.connect", return_value=socket):
            await worker._respond(request)

        completed = [job for job in room.jobs if job.get("terminal")]
        self.assertEqual(completed[-1]["final_text"], "这是查询后的具体车型和价格。")
        self.assertNotEqual(completed[-1]["final_text"], "马上回来告诉你。")
        retry_updates = [
            item["session"] for item in socket.sent
            if item.get("type") == "session.update"
            and "现在只能调用一个最合适" in item["session"].get("instructions", "")
        ]
        self.assertTrue(retry_updates)

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
