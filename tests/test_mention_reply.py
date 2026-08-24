import sys
import unittest
from unittest import mock
from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "s2s" / "hf-realtime-voice"
sys.path.insert(0, str(FRONTEND_DIR))

from mention_reply import (  # noqa: E402
    MentionReplyWorker,
    compact_news_output,
    looks_like_deferred_answer,
    parse_mention,
    plan_research,
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

    def test_realtime_crypto_question_requires_price_and_news(self):
        plan = plan_research("查一下最新国际新闻，为什么比特币涨得这么快")
        self.assertTrue(plan.needs_price)
        self.assertTrue(plan.needs_news)
        self.assertEqual(plan.coin_id, "bitcoin")

    def test_casual_question_does_not_force_research(self):
        self.assertFalse(plan_research("你最喜欢什么动漫").required)

    def test_detects_transition_promise_instead_of_final_answer(self):
        self.assertTrue(looks_like_deferred_answer("好哒，我去查查最新新闻"))
        self.assertFalse(looks_like_deferred_answer("比特币今天上涨，主要与资金流入有关"))

    def test_news_compaction_keeps_multiple_sources(self):
        raw = "\n---\n".join(f"Title: source-{index}\n" + "x" * 2000 for index in range(5))
        compact = compact_news_output(raw)
        self.assertLessEqual(len(compact), 3800)
        self.assertIn("source-0", compact)
        self.assertIn("source-3", compact)


class FakeRoom:
    async def can_bot_reply(self):
        return True


class FakeGateway:
    enabled = True

    def __init__(self, *, exa_fails=False):
        self.exa_fails = exa_fails
        self.calls = []

    async def list_tools(self):
        return [
            {"name": "mcp_coingecko_price"},
            {"name": "mcp_exa_web_search_exa"},
            {"name": "mcp_gdelt_gdelt_search_articles"},
        ]

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "mcp_coingecko_price":
            return '{"bitcoin":{"usd":65000,"cny":468000}}'
        if name == "mcp_exa_web_search_exa" and self.exa_fails:
            return "MCP tool error: rate limited"
        if name == "mcp_exa_web_search_exa":
            return "最新新闻资料"
        return "GDELT 后备资料"


class FailingRss:
    enabled = True

    async def search(self, query):
        raise RuntimeError("RSS unavailable")


class WorkingRss:
    enabled = True

    async def search(self, query):
        return "RSS 最新新闻资料"


class MentionResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_one_proactive_room_topic_can_be_pending(self):
        worker = MentionReplyWorker(FakeRoom(), FakeGateway(), "ws://unused")
        self.assertTrue(worker.enqueue_proactive("讲一条新闻"))
        self.assertFalse(worker.enqueue_proactive("再讲一条新闻"))
        self.assertEqual(len(worker.pending), 1)
        self.assertTrue(worker.pending[0].proactive)

    async def test_prefetches_price_and_exa_news_and_reuses_cache(self):
        gateway = FakeGateway()
        worker = MentionReplyWorker(FakeRoom(), gateway, "ws://unused")
        worker.rss_news = FailingRss()
        plan = plan_research("最新国际新闻，为什么比特币上涨")
        result = await worker._research("最新国际新闻，为什么比特币上涨", plan)
        self.assertIn("65000", result.evidence)
        self.assertIn("最新新闻资料", result.evidence)
        self.assertEqual(len(gateway.calls), 2)

        await worker._research("最新国际新闻，为什么比特币上涨", plan)
        self.assertEqual(len(gateway.calls), 2)

    async def test_falls_back_to_gdelt_when_exa_fails(self):
        gateway = FakeGateway(exa_fails=True)
        worker = MentionReplyWorker(FakeRoom(), gateway, "ws://unused")
        worker.rss_news = FailingRss()
        plan = plan_research("查一下今天的国际新闻")
        result = await worker._research("查一下今天的国际新闻", plan)
        self.assertIn("GDELT 后备资料", result.evidence)
        self.assertEqual(
            [name for name, _ in gateway.calls],
            ["mcp_exa_web_search_exa", "mcp_gdelt_gdelt_search_articles"],
        )

    async def test_prefers_rss_over_news_mcp(self):
        gateway = FakeGateway()
        worker = MentionReplyWorker(FakeRoom(), gateway, "ws://unused")
        worker.rss_news = WorkingRss()
        plan = plan_research("查一下今天的国际新闻")
        result = await worker._research("查一下今天的国际新闻", plan)
        self.assertIn("RSS 最新新闻资料", result.evidence)
        self.assertEqual(gateway.calls, [])

    async def test_analytical_question_enriches_rss_with_search(self):
        gateway = FakeGateway()
        worker = MentionReplyWorker(FakeRoom(), gateway, "ws://unused")
        worker.rss_news = WorkingRss()
        plan = plan_research("为什么比特币今天上涨")
        result = await worker._research("为什么比特币今天上涨", plan)
        self.assertIn("RSS 最新新闻资料", result.evidence)
        self.assertIn("最新新闻资料", result.evidence)
        self.assertEqual(
            [name for name, _ in gateway.calls],
            ["mcp_coingecko_price", "mcp_exa_web_search_exa"],
        )

    async def test_proactive_speech_streams_to_room_and_is_finalized(self):
        class RecordingRoom(FakeRoom):
            def __init__(self): self.items = []
            async def publish_bot_reply(self, **item): self.items.append(dict(item)); return item

        class NoTools:
            enabled = False

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
        worker = MentionReplyWorker(room, NoTools(), "ws://unused")
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
