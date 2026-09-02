import sys
import asyncio
import contextlib
import json
import time
import unittest
from unittest import mock
from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "apps" / "web"
sys.path.insert(0, str(FRONTEND_DIR))

from mention_reply import (  # noqa: E402
    DuplicateReplyDetected,
    MentionReplyWorker,
    looks_like_deferred_answer,
    mention_failure_reply,
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
        self.assertTrue(looks_like_deferred_answer("我正在联网查找相关资料，再核对一下来源呀。"))
        self.assertFalse(looks_like_deferred_answer("比特币今天上涨，主要与资金流入有关"))

    def test_failure_reply_separates_busy_from_blank_answers(self):
        self.assertIn("还在忙", mention_failure_reply(RuntimeError("All session slots are in use")))
        self.assertIn("没答上来", mention_failure_reply(RuntimeError("模型没有生成可播报的最终答案")))
        self.assertNotIn("工具被关闭", mention_failure_reply(RuntimeError("timeout")))

class FakeRoom:
    async def can_bot_reply(self):
        return True

    async def can_start_proactive(self):
        return True

    async def publish_agent_job(self, job, **_kwargs):
        return job


class MentionResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_comment_follows_memory_and_duplicate_reply_is_blocked(self):
        old_reply = "哟，三丰叔又来啦？嘴这么勤快，是不是又来找我补糖呀？"

        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item
            async def participant_memory_context(self, *_args, **kwargs):
                self.exclude_message_id = kwargs.get("exclude_message_id")
                return f"用户：@小麻 你干啥呢\n数字人：{old_reply}"
            async def active_news_context(self, *_args): return ""
            async def latest_assistant_reply(self, *_args, **_kwargs): return old_reply

        class NoTools:
            enabled = False

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
                    json.dumps({"type":"response.audio_transcript.done","transcript":old_reply}, ensure_ascii=False),
                ])
                self.sent = []
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def recv(self): return next(self.events)
            async def send(self, message): self.sent.append(json.loads(message))

        room, socket = RecordingRoom(), FakeWebSocket()
        worker = MentionReplyWorker(room, NoTools(), "ws://unused")
        request = parse_mention({
            "id": "current", "participant_id": "p1", "speaker": "张三丰",
            "text": "@小麻 谁是你叔，我是你哥啊！",
        })
        with mock.patch("mention_reply.websockets.connect", return_value=socket):
            with self.assertRaises(DuplicateReplyDetected):
                await worker._respond(request)

        user_item = next(
            item for item in socket.sent if item.get("type") == "conversation.item.create"
        )
        session = next(item["session"] for item in socket.sent if item.get("type") == "session.update")
        prompt = user_item["item"]["content"][0]["text"]
        self.assertIn("这是公开评论", session["instructions"])
        self.assertNotIn("这是直播间入场欢迎", session["instructions"])
        self.assertEqual(room.exclude_message_id, "current")
        self.assertLess(prompt.index("【历史记忆，仅供理解】"), prompt.index("【当前评论"))
        self.assertTrue(prompt.rstrip().endswith("谁是你叔，我是你哥啊！"))
        self.assertEqual(room.items, [])

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
            def dialogue_web_tool(self):
                raise AssertionError("welcome must not load research tools")
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
        self.assertIn("这是直播间入场欢迎", session["instructions"])
        self.assertNotIn("这是公开评论", session["instructions"])
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

    async def test_empty_room_drops_queued_idle_news(self):
        class EmptyRoom(FakeRoom):
            async def can_start_proactive(self):
                return False

        worker = MentionReplyWorker(EmptyRoom(), mock.Mock(), "ws://unused")
        worker.enqueue_proactive("讲一条新闻")
        worker._run_task = asyncio.create_task(worker._run())
        await asyncio.sleep(0.05)
        worker._run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker._run_task
        self.assertEqual(len(worker.pending), 0)

    async def test_chat_exposes_only_web_search_and_never_lists_mcp(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item
            async def publish_agent_job(self, job, **_kwargs): return job
            async def participant_memory_context(self, *_args, **_kwargs): return ""
            async def active_news_context(self, *_args):
                raise AssertionError("idle chat must not load news")

        class WebOnlyGateway:
            enabled = True
            DISCOVERY_TOOL_NAME = "request_external_capabilities"
            def dialogue_web_tool(self):
                raise AssertionError("idle chat must not load web search")
            async def list_tools(self):
                raise AssertionError("chat must not load the full tool registry")
            async def tools_for_capabilities(self, _capabilities):
                raise AssertionError("chat must not expand extra capabilities")

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
                    '{"type":"response.audio_transcript.done","transcript":"嗯，我在听。"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                ])
                self.sent = []
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def recv(self): return next(self.events)
            async def send(self, message): self.sent.append(json.loads(message))

        room, socket = RecordingRoom(), FakeWebSocket()
        worker = MentionReplyWorker(room, WebOnlyGateway(), "ws://unused")
        request = parse_mention({
            "id": "chat", "participant_id": "p1", "speaker": "张三丰", "text": "@小麻 哦",
        })
        with mock.patch("mention_reply.DIALOGUE_TOOLS_ENABLED", True), mock.patch(
            "mention_reply.websockets.connect", return_value=socket
        ):
            await worker._respond(request)
        session = next(item["session"] for item in socket.sent if item.get("type") == "session.update")
        self.assertEqual(session.get("tools") or [], [])
        self.assertIn("简单闲聊", session["instructions"])
        self.assertNotIn("smart_web_search", session["instructions"])
        self.assertNotIn("request_external_capabilities", session["instructions"])
        self.assertEqual(room.items[-1]["text"], "嗯，我在听。")

    async def test_chat_web_search_runs_with_progress(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item
            async def publish_agent_job(self, job, **_kwargs): self.items.append(dict(job)); return job
            async def participant_memory_context(self, *_args, **_kwargs): return ""
            async def active_news_context(self, *_args): return ""

        class WebOnlyGateway:
            enabled = True
            DISCOVERY_TOOL_NAME = "request_external_capabilities"
            def __init__(self): self.calls = []
            def dialogue_web_tool(self):
                return {
                    "type": "function", "name": "smart_web_search", "description": "联网查询",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    "progress_text": "我先帮你查清楚呀。",
                }
            async def list_tools(self):
                raise AssertionError("chat must not load the full tool registry")
            async def call(self, name, arguments):
                self.calls.append((name, arguments))
                await asyncio.sleep(0.02)
                return json.dumps({"result": "可信资料"}, ensure_ascii=False)

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
                    '{"type":"response.function_call_arguments.done","name":"smart_web_search","arguments":"{\\"query\\":\\"查证一下\\"}","call_id":"c1"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                    '{"type":"response.audio_transcript.done","transcript":"这是核对后的最终答案。"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                ])
                self.sent = []
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def recv(self): return next(self.events)
            async def send(self, message): self.sent.append(json.loads(message))

        room, gateway, socket = RecordingRoom(), WebOnlyGateway(), FakeWebSocket()
        worker = MentionReplyWorker(room, gateway, "ws://unused")
        request = parse_mention({"id": "tools", "participant_id": "p1", "speaker": "观众", "text": "@小麻 查证一下"})
        worker._speak_exact = mock.AsyncMock()
        with mock.patch("mention_reply.DIALOGUE_TOOLS_ENABLED", True), mock.patch(
            "mention_reply.websockets.connect", return_value=socket
        ):
            await worker._respond(request)
        self.assertEqual([name for name, _ in gateway.calls], ["smart_web_search"])
        worker._speak_exact.assert_awaited()
        self.assertTrue(any(item.get("text") == "我去看一眼，马上回你。" for item in room.items))
        self.assertFalse(any(item.get("text") == "我先帮你查清楚呀。" for item in room.items))
        self.assertTrue(any(item.get("text") == "这是核对后的最终答案。" for item in room.items))
        session_updates = [item["session"] for item in socket.sent if item.get("type") == "session.update"]
        self.assertEqual(session_updates[0].get("tool_choice"), "auto")
        self.assertEqual([tool["name"] for tool in session_updates[0].get("tools") or []], ["smart_web_search"])
        self.assertTrue(any(item.get("tool_choice") == "auto" for item in session_updates))
        self.assertTrue(any("中文口语" in item.get("instructions", "") for item in session_updates))

    async def test_news_mention_uses_prefetched_headlines_without_tools(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item
            async def publish_agent_job(self, job, **_kwargs): self.items.append(dict(job)); return job
            async def participant_memory_context(self, *_args, **_kwargs): return ""
            async def active_news_context(self, *_args): return ""

        class PrefetchGateway:
            enabled = True
            DISCOVERY_TOOL_NAME = "request_external_capabilities"
            def __init__(self):
                self.queries = []
            async def prefetch_spoken_evidence(self, query):
                self.queries.append(query)
                return "刚才查到的最新资讯：\n1. 某国发布了新政策\n   摘要：今天正式落地。"
            def dialogue_web_tool(self):
                raise AssertionError("prefetched news must not expose search tools")
            async def call(self, *_args, **_kwargs):
                raise AssertionError("prefetched news must not call tools")

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
                    json.dumps({
                        "type": "response.audio_transcript.done",
                        "transcript": "刚看到一条，某国今天落地了新政策。",
                    }, ensure_ascii=False),
                    '{"type":"response.done","response":{"status":"completed"}}',
                ])
                self.sent = []
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def recv(self): return next(self.events)
            async def send(self, message): self.sent.append(json.loads(message))

        room, gateway, socket = RecordingRoom(), PrefetchGateway(), FakeWebSocket()
        worker = MentionReplyWorker(room, gateway, "ws://unused")
        worker._speak_exact = mock.AsyncMock()
        request = parse_mention({
            "id": "news", "participant_id": "p1", "speaker": "张三丰",
            "text": "@小麻 看看最新新闻",
        })
        with mock.patch("mention_reply.DIALOGUE_TOOLS_ENABLED", True), mock.patch(
            "mention_reply.websockets.connect", return_value=socket
        ):
            await worker._respond(request)
        self.assertEqual(gateway.queries, ["看看最新新闻"])
        worker._speak_exact.assert_awaited()
        self.assertEqual(worker._speak_exact.await_args.args[1], "我翻一下今天的，马上说。")
        self.assertTrue(any(item.get("text") == "我翻一下今天的，马上说。" for item in room.items))
        user_item = next(
            item["item"]["content"][0]["text"]
            for item in socket.sent
            if item.get("type") == "conversation.item.create"
            and item.get("item", {}).get("role") == "user"
        )
        self.assertIn("【已查到的资料】", user_item)
        self.assertIn("某国发布了新政策", user_item)
        session = next(item["session"] for item in socket.sent if item.get("type") == "session.update")
        self.assertEqual(session.get("tools") or [], [])
        self.assertIn("已经查到", session["instructions"])
        self.assertFalse(any("正在联网查找" in str(item.get("text") or "") for item in room.items))
        self.assertTrue(any("某国今天落地了新政策" in str(item.get("text") or "") for item in room.items))

    async def test_failed_prefetch_speaks_a_lookup_failure_instead_of_chat(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item
            async def publish_agent_job(self, job, **_kwargs): self.items.append(dict(job)); return job

        class FailingPrefetch:
            enabled = True
            async def prefetch_spoken_evidence(self, _query):
                raise RuntimeError("tavily: ConnectTimeout")
            def dialogue_web_tool(self):
                raise AssertionError("failed prefetch must not wait for another tool call")

        class FakeWebSocket:
            def __init__(self):
                self.events = iter(['{"type":"session.created"}'])
                self.sent = []
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def recv(self): return next(self.events)
            async def send(self, message): self.sent.append(json.loads(message))

        room, socket = RecordingRoom(), FakeWebSocket()
        worker = MentionReplyWorker(room, FailingPrefetch(), "ws://unused")
        worker._speak_exact = mock.AsyncMock()
        request = parse_mention({
            "id": "news-fail", "participant_id": "p1", "speaker": "张三丰",
            "text": "@小麻 看看最新有什么新闻说来听听",
        })
        with mock.patch("mention_reply.DIALOGUE_TOOLS_ENABLED", True), mock.patch(
            "mention_reply.websockets.connect", return_value=socket
        ):
            await worker._respond(request)
        texts = [str(item.get("text") or "") for item in room.items]
        self.assertTrue(any("我翻一下今天的，马上说。" in text for text in texts))
        self.assertTrue(any("没翻到" in text for text in texts))
        spoken = [call.args[1] for call in worker._speak_exact.await_args_list]
        self.assertEqual(spoken[0], "我翻一下今天的，马上说。")
        self.assertIn("没翻到", spoken[-1])
        self.assertFalse(any(item.get("type") == "response.create" and not (
            isinstance(item.get("response"), dict)
            and item["response"].get("metadata", {}).get("client_purpose") == "tool_progress"
        ) for item in socket.sent))

    async def test_empty_answer_after_search_speaks_a_retry_line(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item
            async def publish_agent_job(self, job, **_kwargs): return job
            async def participant_memory_context(self, *_args, **_kwargs): return ""
            async def active_news_context(self, *_args): return ""

        class WebOnlyGateway:
            enabled = True
            DISCOVERY_TOOL_NAME = "request_external_capabilities"
            def dialogue_web_tool(self):
                return {
                    "type": "function", "name": "smart_web_search", "description": "联网查询",
                    "parameters": {"type": "object"}, "progress_text": "我先帮你查清楚呀。",
                }
            async def call(self, _name, _arguments):
                return "工具调用失败：ConnectTimeout"

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
                    '{"type":"response.function_call_arguments.done","name":"smart_web_search","arguments":"{\\"query\\":\\"今天新闻\\"}","call_id":"c1"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                ])
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def recv(self): return next(self.events)
            async def send(self, _message): pass

        room = RecordingRoom()
        worker = MentionReplyWorker(room, WebOnlyGateway(), "ws://unused")
        worker._speak_exact = mock.AsyncMock()
        request = parse_mention({
            "id": "news", "participant_id": "p1", "speaker": "张三丰",
            "text": "@小麻 你查一下今天有啥好玩的新闻不",
        })
        with mock.patch("mention_reply.DIALOGUE_TOOLS_ENABLED", True), mock.patch(
            "mention_reply.websockets.connect", return_value=FakeWebSocket()
        ):
            await worker._respond(request)
        self.assertTrue(any("没把这条查完整" in str(item.get("text") or "") for item in room.items))
        self.assertFalse(any("没答上来" in str(item.get("text") or "") for item in room.items))

    async def test_english_search_summary_is_rejected_for_chinese_chat(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item
            async def publish_agent_job(self, job, **_kwargs): return job
            async def participant_memory_context(self, *_args, **_kwargs): return ""
            async def active_news_context(self, *_args): return ""

        class WebOnlyGateway:
            enabled = True
            DISCOVERY_TOOL_NAME = "request_external_capabilities"
            def dialogue_web_tool(self):
                return {
                    "type": "function", "name": "smart_web_search", "description": "联网查询",
                    "parameters": {"type": "object"}, "progress_text": "我先帮你查清楚呀。",
                }
            async def call(self, _name, _arguments):
                return '{"results":[{"title":"Seal becomes celebrity"}]}'

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
                    '{"type":"response.function_call_arguments.done","name":"smart_web_search","arguments":"{\\"query\\":\\"今天新闻\\"}","call_id":"c1"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                    '{"type":"response.audio_transcript.done","transcript":"Based on the search results provided, here is a summary of the latest news."}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                    '{"type":"response.audio_transcript.done","transcript":"今天有条挺好玩的，一只海豹上了路。"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                ])
                self.sent = []
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def recv(self): return next(self.events)
            async def send(self, message): self.sent.append(json.loads(message))

        room = RecordingRoom()
        worker = MentionReplyWorker(room, WebOnlyGateway(), "ws://unused")
        worker._speak_exact = mock.AsyncMock()
        request = parse_mention({
            "id": "en", "participant_id": "p1", "speaker": "张三丰",
            "text": "@小麻 今天有啥有趣的补充新闻",
        })
        with mock.patch("mention_reply.DIALOGUE_TOOLS_ENABLED", True), mock.patch(
            "mention_reply.websockets.connect", return_value=FakeWebSocket()
        ):
            await worker._respond(request)
        finals = [item["text"] for item in room.items if item.get("text") and not item.get("partial")]
        self.assertTrue(any("海豹" in text for text in finals))
        self.assertFalse(any("Based on the search results" in text for text in finals))

    async def test_leftover_discovery_call_does_not_expand_tools(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item
            async def publish_agent_job(self, job, **_kwargs): return job
            async def participant_memory_context(self, *_args, **_kwargs): return ""
            async def active_news_context(self, *_args): return ""

        class WebOnlyGateway:
            enabled = True
            DISCOVERY_TOOL_NAME = "request_external_capabilities"
            def dialogue_web_tool(self):
                return {
                    "type": "function", "name": "smart_web_search", "description": "联网查询",
                    "parameters": {"type": "object"}, "progress_text": "我先帮你查清楚呀。",
                }
            async def list_tools(self):
                raise AssertionError("leftover discovery must not list MCP tools")
            async def tools_for_capabilities(self, _capabilities):
                raise AssertionError("leftover discovery must not expand capabilities")

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
                    '{"type":"response.function_call_arguments.done","name":"request_external_capabilities","arguments":"{\\"capabilities\\":[\\"web\\",\\"news\\",\\"market\\"]}","call_id":"d1"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                    '{"type":"response.audio_transcript.done","transcript":"我在，你说。"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                ])
                self.sent = []
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def recv(self): return next(self.events)
            async def send(self, message): self.sent.append(json.loads(message))

        room, socket = RecordingRoom(), FakeWebSocket()
        worker = MentionReplyWorker(room, WebOnlyGateway(), "ws://unused")
        request = parse_mention({"id": "left", "participant_id": "p1", "speaker": "观众", "text": "@小麻 什么玩意"})
        with mock.patch("mention_reply.DIALOGUE_TOOLS_ENABLED", True), mock.patch(
            "mention_reply.websockets.connect", return_value=socket
        ):
            await worker._respond(request)
        self.assertEqual(room.items[-1]["text"], "我在，你说。")
        outputs = [
            item["item"]["output"]
            for item in socket.sent
            if item.get("type") == "conversation.item.create"
            and item.get("item", {}).get("type") == "function_call_output"
        ]
        self.assertTrue(outputs)
        self.assertIn("直接自然回答", outputs[0])
        self.assertNotIn("local_rss_news", outputs[0])
        self.assertNotIn("mcp_", outputs[0])

    async def test_deferred_lookup_promise_must_call_web_search(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.jobs = []
            async def publish_bot_reply(self, **item): return item
            async def publish_agent_job(self, job, **_kwargs): self.jobs.append(dict(job)); return job
            async def participant_memory_context(self, *_args, **_kwargs): return ""
            async def active_news_context(self, *_args): return ""

        class WebOnlyGateway:
            enabled = True
            DISCOVERY_TOOL_NAME = "request_external_capabilities"
            def dialogue_web_tool(self):
                return {
                    "type": "function", "name": "smart_web_search", "description": "联网查询",
                    "parameters": {"type": "object"}, "progress_text": "我先帮你查清楚呀。",
                }
            async def list_tools(self):
                raise AssertionError("chat must not load the full tool registry")
            async def call(self, _name, _arguments): return '{"result":"小米汽车可信资料"}'

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
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
        worker = MentionReplyWorker(room, WebOnlyGateway(), "ws://unused")
        worker._speak_exact = mock.AsyncMock()
        request = parse_mention({
            "id": "car", "participant_id": "p1", "speaker": "观众",
            "text": "@小麻 小米最新的汽车是什么，多少钱",
        })
        with mock.patch("mention_reply.DIALOGUE_TOOLS_ENABLED", True), mock.patch(
            "mention_reply.websockets.connect", return_value=socket
        ):
            await worker._respond(request)

        completed = [job for job in room.jobs if job.get("terminal")]
        self.assertEqual(completed[-1]["final_text"], "这是查询后的具体车型和价格。")
        self.assertNotEqual(completed[-1]["final_text"], "马上回来告诉你。")
        retry_updates = [
            item["session"] for item in socket.sent
            if item.get("type") == "session.update"
            and "立刻调用 smart_web_search" in item["session"].get("instructions", "")
        ]
        self.assertTrue(retry_updates)

    async def test_proactive_speech_streams_to_room_and_is_finalized(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item

        class ToolsMustNotBeLoaded:
            enabled = True
            def dialogue_web_tool(self):
                raise AssertionError("proactive speech must use its preloaded evidence")
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

    async def test_welcome_and_news_use_the_live_chat_socket(self):
        seen: list[str] = []

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"session.created"}',
                    '{"type":"response.audio_transcript.done","transcript":"诶，三丰。"}',
                    '{"type":"response.done","response":{"status":"completed"}}',
                ])

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def recv(self):
                return next(self.events)

            async def send(self, _message):
                pass

        def connect(url, **_kwargs):
            seen.append(url)
            return FakeWebSocket()

        class RecordingRoom(FakeRoom):
            def __init__(self):
                self.items = []

            async def publish_bot_reply(self, **item):
                self.items.append(dict(item))
                return item

        worker = MentionReplyWorker(
            RecordingRoom(), mock.Mock(enabled=False), "ws://s2s/v1/realtime"
        )
        with mock.patch("mention_reply.websockets.connect", side_effect=connect):
            worker.enqueue_welcome(participant_id="p1", speaker="三丰")
            await worker._respond(worker.pending.popleft())
            worker.enqueue_proactive("讲新闻")
            await worker._respond(worker.pending.popleft())
        self.assertEqual(seen, ["ws://s2s/v1/realtime", "ws://s2s/v1/realtime"])

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

    async def test_speak_exact_marks_the_wait_beat_as_tool_progress(self):
        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    '{"type":"response.done","response":{"status":"completed"}}',
                ])
                self.sent = []

            async def recv(self):
                return next(self.events)

            async def send(self, message):
                self.sent.append(json.loads(message))

        worker = MentionReplyWorker(FakeRoom(), mock.Mock(enabled=False), "ws://unused")
        socket = FakeWebSocket()
        await worker._speak_exact(socket, "我翻一下今天的，马上说。")
        create = next(item for item in socket.sent if item.get("type") == "response.create")
        self.assertEqual(create["response"]["metadata"]["client_purpose"], "tool_progress")
        user_item = next(
            item for item in socket.sent if item.get("type") == "conversation.item.create"
        )
        self.assertIn("我翻一下今天的，马上说。", user_item["item"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
