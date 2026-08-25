import json
import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "s2s" / "hf-realtime-voice"
sys.path.insert(0, str(FRONTEND_DIR))

from smart_search import SearchHit, SmartSearchGateway  # noqa: E402


class SmartSearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env = mock.patch.dict(os.environ, {
            "TAVILY_API_KEY": "tvly-test",
            "EXA_API_KEY": "exa-test",
            "JINA_READER_ENABLED": "1",
            "SEARXNG_URL": "",
        }, clear=False)
        self.env.start()
        self.gateway = SmartSearchGateway()

    async def asyncTearDown(self):
        await self.gateway.close()
        self.env.stop()

    async def test_tavily_is_primary_and_output_is_normalized(self):
        hit = SearchHit("标题", "https://example.com/a", "摘要", "Tavily", score=0.9)
        with mock.patch.object(self.gateway, "_tavily", mock.AsyncMock(return_value=[hit])) as tavily, \
             mock.patch.object(self.gateway, "_exa", mock.AsyncMock()) as exa:
            output = json.loads(await self.gateway.search("今天的科技新闻", topic="news"))
        self.assertEqual(output["sources"], ["tavily"])
        self.assertEqual(output["results"][0]["title"], "标题")
        tavily.assert_awaited_once()
        exa.assert_not_awaited()

    async def test_exa_is_used_when_tavily_fails(self):
        hit = SearchHit("备用结果", "https://example.org/b", "可信摘要", "Exa")
        with mock.patch.object(self.gateway, "_tavily", mock.AsyncMock(side_effect=RuntimeError("busy"))), \
             mock.patch.object(self.gateway, "_exa", mock.AsyncMock(return_value=[hit])):
            output = json.loads(await self.gateway.search("一个新的问题"))
        self.assertEqual(output["sources"], ["exa"])
        self.assertEqual(output["results"][0]["source"], "Exa")

    async def test_cache_avoids_spending_provider_quota_twice(self):
        hit = SearchHit("缓存结果", "https://example.net/c", "摘要", "Tavily")
        provider = mock.AsyncMock(return_value=[hit])
        with mock.patch.object(self.gateway, "_tavily", provider):
            first = await self.gateway.search("重复查询")
            second = await self.gateway.search("重复查询")
        self.assertEqual(first, second)
        self.assertEqual(provider.await_count, 1)

    async def test_evidence_search_queries_tavily_and_exa_concurrently(self):
        tavily_hit = SearchHit("原因一", "https://one.example/a", "资金流入", "Tavily")
        exa_hit = SearchHit("原因二", "https://two.example/b", "宏观流动性", "Exa")
        tavily = mock.AsyncMock(return_value=[tavily_hit])
        exa = mock.AsyncMock(return_value=[exa_hit])
        with mock.patch.object(self.gateway, "_tavily", tavily), \
             mock.patch.object(self.gateway, "_exa", exa):
            output = json.loads(await self.gateway.search_all("比特币上涨原因", topic="news"))
        self.assertEqual(output["sources"], ["tavily", "exa"])
        self.assertEqual({item["title"] for item in output["results"]}, {"原因一", "原因二"})
        tavily.assert_awaited_once()
        exa.assert_awaited_once()

    async def test_evidence_search_keeps_fast_source_when_another_times_out(self):
        self.gateway.evidence_budget_seconds = 0.02
        fast_hit = SearchHit("及时结果", "https://fast.example/a", "可核实摘要", "Tavily")

        async def slow_provider(*_args, **_kwargs):
            await asyncio.sleep(1)
            return [SearchHit("迟到结果", "https://slow.example/b", "摘要", "Exa")]

        with mock.patch.object(self.gateway, "_tavily", mock.AsyncMock(return_value=[fast_hit])), \
             mock.patch.object(self.gateway, "_exa", side_effect=slow_provider):
            output = json.loads(await self.gateway.search_all("时效问题"))
        self.assertEqual(output["sources"], ["tavily"])
        self.assertEqual(output["results"][0]["title"], "及时结果")

    async def test_private_and_malformed_urls_are_removed(self):
        self.assertEqual(self.gateway._valid_public_url("http://127.0.0.1/private"), "")
        self.assertEqual(self.gateway._valid_public_url("file:///etc/passwd"), "")
        self.assertEqual(
            self.gateway._valid_public_url("https://example.com/page"),
            "https://example.com/page",
        )

    async def test_reader_output_is_structured_and_does_not_leak_raw_urls(self):
        response = mock.Mock()
        response.text = (
            "Title: 示例新闻\nURL Source: https://example.com/a\n"
            "Published Time: 2026-08-24\nMarkdown Content:\n"
            "[正文链接](https://example.com/detail) 内容。"
        )
        response.raise_for_status.return_value = None
        with mock.patch.object(self.gateway._http, "get", mock.AsyncMock(return_value=response)), \
             mock.patch("smart_search.socket.getaddrinfo", return_value=[]):
            output = json.loads(await self.gateway.fetch("https://example.com/a"))
        self.assertEqual(output["title"], "示例新闻")
        self.assertEqual(output["source"], "example.com")
        self.assertNotIn("https://", output["content"])
        self.assertIn("正文链接", output["content"])


if __name__ == "__main__":
    unittest.main()
