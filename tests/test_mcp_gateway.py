import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import httpx


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "s2s" / "hf-realtime-voice"
sys.path.insert(0, str(FRONTEND_DIR))

from mcp_gateway import McpGateway, _parse_rpc_response, _tool_output  # noqa: E402


class McpGatewayTests(unittest.TestCase):
    def test_discovery_tool_includes_voice_only_vision_capability(self):
        capabilities = McpGateway.discovery_tool()["parameters"]["properties"]["capabilities"]
        self.assertIn("vision", capabilities["items"]["enum"])

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


if __name__ == "__main__":
    unittest.main()
