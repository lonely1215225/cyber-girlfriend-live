import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import httpx


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "apps" / "web"
sys.path.insert(0, str(FRONTEND_DIR))

from mcp_gateway import McpGateway, _parse_rpc_response, _tool_output  # noqa: E402


class McpGatewayTests(unittest.TestCase):
    def test_discovery_tool_includes_voice_only_vision_capability(self):
        capabilities = McpGateway.discovery_tool()["parameters"]["properties"]["capabilities"]
        self.assertIn("vision", capabilities["items"]["enum"])
        self.assertNotIn("conversation", capabilities["items"]["enum"])
        self.assertIn("Never call it for greetings", McpGateway.discovery_tool()["description"])

    def test_dialogue_web_tool_is_search_only_without_listing_mcp(self):
        with mock.patch.dict(
            "os.environ",
            {"MCP_ENABLED": "0", "TAVILY_API_KEY": "tvly-test", "EXA_API_KEY": "", "SEARXNG_URL": ""},
        ):
            gateway = McpGateway()
        gateway.list_tools = mock.AsyncMock(side_effect=AssertionError("must not list MCP"))
        try:
            tool = gateway.dialogue_web_tool()
            self.assertIsNotNone(tool)
            self.assertEqual(tool["name"], "smart_web_search")
            self.assertEqual(tool["source"], "smart-search")
        finally:
            pass

    def test_dialogue_web_tool_missing_without_search_keys(self):
        with mock.patch.dict(
            "os.environ",
            {"MCP_ENABLED": "0", "TAVILY_API_KEY": "", "EXA_API_KEY": "", "SEARXNG_URL": ""},
        ):
            gateway = McpGateway()
        self.assertIsNone(gateway.dialogue_web_tool())

    def test_parses_streamable_http_sse_response(self):
        payload = {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "search"}]}}
        response = httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=f"event: message\ndata: {json.dumps(payload)}\n\n",
        )
        self.assertEqual(_parse_rpc_response(response), payload)

    def test_formats_text_and_structured_tool_results(self):
        self.assertEqual(
            _tool_output({"content": [{"type": "text", "text": "实时结果"}]}),
            "实时结果",
        )
        structured = _tool_output({"structuredContent": {"price": 123}})
        self.assertIn('"price": 123', structured)

    def test_marks_mcp_errors_for_the_model(self):
        output = _tool_output(
            {"isError": True, "content": [{"type": "text", "text": "rate limited"}]}
        )
        self.assertEqual(output, "MCP tool error: rate limited")


class LocalRssToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_rss_tool_is_always_exposed_and_callable(self):
        class FakeRss:
            enabled = True

            async def query_topics(self, **kwargs):
                return f"{kwargs['category']}:{kwargs['source']}:{kwargs['query']}"

        gateway = McpGateway()
        gateway.clients = {}
        gateway.rss_news = FakeRss()
        tools = await gateway.list_tools()
        self.assertIn("local_rss_news", {tool["name"] for tool in tools})
        output = await gateway.call(
            "local_rss_news",
            {"category": "科技", "source": "IT之家", "query": "今天的新闻"},
        )
        self.assertEqual(output, "科技:IT之家:今天的新闻")

    async def test_smart_search_tools_are_exposed_when_configured(self):
        with mock.patch.dict(
            "os.environ", {"MCP_ENABLED": "0", "TAVILY_API_KEY": "tvly-test"}
        ):
            gateway = McpGateway()
        gateway.rss_news.enabled = False
        try:
            tools = await gateway.list_tools()
            names = {tool["name"] for tool in tools}
            self.assertIn("smart_web_search", names)
            self.assertIn("smart_web_fetch", names)
        finally:
            await gateway.close()

    async def test_web_search_falls_back_to_rss_when_providers_timeout(self):
        gateway = McpGateway()
        gateway.smart_search.search = mock.AsyncMock(side_effect=RuntimeError("tavily: ConnectTimeout"))
        gateway.rss_news.enabled = True
        gateway.rss_news.spoken_brief = mock.AsyncMock(return_value="刚才查到的最新资讯：一条好玩的新闻")
        gateway.rss_news.query_topics = mock.AsyncMock(return_value="RSS 最新资讯：一条好玩的新闻")
        try:
            output = await gateway.call("smart_web_search", {"query": "今天有啥好玩的新闻"})
            self.assertIn("好玩的新闻", output)
            self.assertEqual(gateway.smart_search.search.await_args.kwargs["topic"], "news")
        finally:
            await gateway.close()

    async def test_prefetch_uses_configured_search_before_rss(self):
        gateway = McpGateway()
        gateway.rss_news.enabled = True
        gateway.smart_search.tavily_key = "tvly-test"
        gateway.rss_news.spoken_brief = mock.AsyncMock(
            side_effect=AssertionError("configured Tavily/Exa must be tried first")
        )
        gateway.smart_search.search = mock.AsyncMock(return_value=json.dumps({
            "results": [{"title": "某国发布了新政策", "source": "Tavily", "snippet": "今天落地"}],
        }, ensure_ascii=False))
        try:
            news = await gateway.prefetch_spoken_evidence("看看最新有什么新闻说来听听")
            self.assertIn("某国发布了新政策", news)
            self.assertEqual(gateway.smart_search.search.await_args.args[0], "今天国内外热点新闻")
            self.assertEqual(gateway.smart_search.search.await_args.kwargs["topic"], "news")
            self.assertTrue(gateway.smart_search.search.await_args.kwargs["ignore_circuit"])
            gateway.rss_news.spoken_brief.assert_not_awaited()
        finally:
            await gateway.close()

        gateway = McpGateway()
        gateway.rss_news.enabled = True
        gateway.smart_search.tavily_key = "tvly-test"
        gateway.rss_news.spoken_brief = mock.AsyncMock(
            return_value="刚才查到的最新资讯：\n1. 备用头条"
        )
        gateway.smart_search.search = mock.AsyncMock(side_effect=RuntimeError("tavily: ConnectTimeout"))
        try:
            with self.assertRaises(RuntimeError):
                await gateway.prefetch_spoken_evidence("看看最新新闻")
            gateway.rss_news.spoken_brief.assert_not_awaited()
        finally:
            await gateway.close()

        gateway = McpGateway()
        gateway.rss_news.enabled = True
        gateway.smart_search.tavily_key = ""
        gateway.smart_search.exa_key = ""
        gateway.smart_search.searxng_url = ""
        gateway.rss_news.spoken_brief = mock.AsyncMock(
            return_value="刚才查到的最新资讯：\n1. 备用头条"
        )
        gateway.smart_search.search = mock.AsyncMock(
            side_effect=AssertionError("paid search must stay unused without keys")
        )
        try:
            news = await gateway.prefetch_spoken_evidence("看看最新新闻")
            self.assertIn("备用头条", news)
            gateway.rss_news.spoken_brief.assert_awaited_once()
            gateway.smart_search.search.assert_not_awaited()
        finally:
            await gateway.close()

        gateway = McpGateway()
        gateway.rss_news.enabled = False
        gateway.smart_search.tavily_key = "tvly-test"
        gateway.smart_search.search = mock.AsyncMock(return_value=json.dumps({
            "results": [{"title": "比特币现价", "source": "Tavily", "snippet": "上涨"}],
        }, ensure_ascii=False))
        try:
            spoken = await gateway.prefetch_spoken_evidence("帮我查一下现在比特币多少钱")
            self.assertIn("刚才查到的资料", spoken)
            self.assertIn("比特币现价", spoken)
            self.assertEqual(gateway.smart_search.search.await_args.kwargs["topic"], "general")
        finally:
            await gateway.close()

    async def test_live_headlines_use_search_when_configured(self):
        gateway = McpGateway()
        gateway.rss_news.enabled = True
        gateway.smart_search.tavily_key = "tvly-test"
        gateway.rss_news.latest_topics = mock.AsyncMock(
            side_effect=AssertionError("configured search must not wait on RSS")
        )
        gateway.smart_search.search = mock.AsyncMock(return_value=json.dumps({
            "results": [{"title": "今日头条", "source": "Tavily", "snippet": "摘要"}],
        }, ensure_ascii=False))
        try:
            source, payload = await gateway.fetch_live_headlines()
            self.assertEqual(source, "search")
            self.assertIn("今日头条", payload)
            gateway.rss_news.latest_topics.assert_not_awaited()
        finally:
            await gateway.close()

        gateway = McpGateway()
        gateway.rss_news.enabled = True
        gateway.smart_search.tavily_key = ""
        gateway.smart_search.exa_key = ""
        gateway.smart_search.searxng_url = ""
        gateway.rss_news.latest_topics = mock.AsyncMock(return_value="RSS 最新资讯：一条备用")
        try:
            source, payload = await gateway.fetch_live_headlines()
            self.assertEqual(source, "rss")
            self.assertIn("备用", payload)
        finally:
            await gateway.close()


if __name__ == "__main__":
    unittest.main()
