import json
import sys
import unittest
from pathlib import Path

import httpx


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "s2s" / "hf-realtime-voice"
sys.path.insert(0, str(FRONTEND_DIR))

from mcp_gateway import _parse_rpc_response, _tool_output  # noqa: E402


class McpGatewayTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
