import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proxy"))

from ollama_thinkless import (  # noqa: E402
    ModelOutputSanitizer,
    _is_exact_speech_request,
    _is_fast_conversation_followup,
    _is_fast_discovery_turn,
    _is_fast_external_planning,
    _needs_reliable_external_route,
    _is_room_welcome_request,
    clean_model_output,
    completed_response,
    local_compaction,
    request_messages,
    to_messages,
    to_ollama_tools,
)


class ModelOutputSanitizerTests(unittest.TestCase):
    def test_responses_top_level_instructions_become_system_message(self):
        messages = request_messages({
            "instructions": "你叫小麻，只输出自然口语。",
            "input": [{"role": "user", "content": "你是谁"}],
        })
        self.assertEqual(messages[0], {
            "role": "system", "content": "你叫小麻，只输出自然口语。",
        })
        self.assertEqual(messages[1], {"role": "user", "content": "你是谁"})

    def test_only_explicit_external_intent_forces_the_reliable_router(self):
        self.assertFalse(_needs_reliable_external_route([
            {"role": "user", "content": "喂，听得见我说话吗？"},
        ]))
        self.assertFalse(_needs_reliable_external_route([
            {"role": "user", "content": "你为什么对我这么好呀？"},
        ]))
        self.assertTrue(_needs_reliable_external_route([
            {"role": "user", "content": "帮我查一下现在比特币多少钱"},
        ]))
        self.assertTrue(_needs_reliable_external_route([
            {"role": "user", "content": "为什么比特币最近涨这么多"},
        ]))

    def test_only_the_initial_discovery_turn_uses_fast_local_model(self):
        payload = {
            "tools": [{"type": "function", "name": "request_external_capabilities"}],
            "input": [{"role": "user", "content": "晚上好"}],
        }
        self.assertTrue(_is_fast_discovery_turn(payload))
        payload["input"].append({"type": "function_call_output", "output": "enabled"})
        self.assertFalse(_is_fast_discovery_turn(payload))
        payload["tools"].append({"type": "function", "name": "smart_web_search"})
        self.assertFalse(_is_fast_discovery_turn(payload))

    def test_conversation_route_followup_stays_local_without_research_tools(self):
        payload = {
            "tools": [],
            "input": [{
                "type": "function_call_output",
                "output": '{"enabled":[],"route":"conversation_fast"}',
            }],
        }
        self.assertTrue(_is_fast_conversation_followup(payload))
        payload["tools"] = [{"type": "function", "name": "smart_web_search"}]
        self.assertFalse(_is_fast_conversation_followup(payload))

    def test_external_first_tool_planning_stays_local(self):
        payload = {
            "tools": [{"type": "function", "name": "smart_web_search"}],
            "input": [{
                "type": "function_call_output",
                "output": '{"route":"external_research","enabled":["web"]}',
            }],
        }
        self.assertTrue(_is_fast_external_planning(payload))
        payload["input"][-1]["output"] = "search evidence"
        self.assertFalse(_is_fast_external_planning(payload))

    def test_fixed_tts_readout_stays_on_local_provider(self):
        messages = [
            {"role": "system", "content": "只逐字朗读用户提供的文字，不要改写。"},
            {"role": "user", "content": "逐字朗读：我查到了。"},
        ]
        self.assertTrue(_is_exact_speech_request(messages))
        self.assertFalse(_is_exact_speech_request([{"role": "user", "content": "现在多少钱？"}]))

    def test_room_welcome_uses_fast_local_provider(self):
        self.assertTrue(_is_room_welcome_request([
            {"role": "system", "content": "你现在是直播间入场欢迎生成器。"},
            {"role": "user", "content": "欢迎林清欢"},
        ]))
        self.assertFalse(_is_room_welcome_request([{"role": "user", "content": "普通聊天"}]))

    def test_translates_responses_api_tools_for_ollama(self):
        tools = to_ollama_tools(
            [{"type": "function", "name": "search", "description": "Search", "parameters": {"type": "object"}}]
        )
        self.assertEqual(tools[0]["function"]["name"], "search")
        self.assertEqual(tools[0]["function"]["parameters"], {"type": "object"})

    def test_preserves_tool_call_and_output_in_ollama_history(self):
        messages = to_messages(
            [
                {"type": "function_call", "name": "search", "call_id": "call_1", "arguments": '{"q":"新闻"}'},
                {"type": "function_call_output", "call_id": "call_1", "output": "查到一条新闻"},
            ]
        )
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["arguments"], {"q": "新闻"})
        self.assertEqual(messages[1], {"role": "tool", "content": "查到一条新闻"})

    def test_completed_response_contains_function_call(self):
        response = completed_response(
            "model",
            "",
            1,
            2,
            [{"id": "call_native", "function": {"name": "search", "arguments": {"q": "ETH"}}}],
        )
        self.assertEqual(response["output"][0]["type"], "function_call")
        self.assertEqual(response["output"][0]["arguments"], '{"q":"ETH"}')

    def test_removes_complete_reasoning_blocks(self):
        self.assertEqual(clean_model_output("<think>秘密推理</think>最终回答"), "最终回答")
        self.assertEqual(clean_model_output("<analysis>hidden</analysis>答案"), "答案")

    def test_handles_tags_split_across_stream_chunks(self):
        cleaner = ModelOutputSanitizer()
        chunks = ["<thi", "nk>不能显示", "</th", "ink>我先", "上香吧"]
        visible = "".join(cleaner.feed(chunk) for chunk in chunks)
        visible += cleaner.feed("", final=True)
        self.assertEqual(visible, "我先上香吧")

    def test_preserves_normal_angle_bracket_text(self):
        self.assertEqual(clean_model_output("价格 < 10，答案正常"), "价格 < 10，答案正常")

    def test_local_compaction_is_extractive_and_skips_hidden_prompts(self):
        prompt = """Summarize the following conversation.
--- CONVERSATION START ---
User: 现在主动欢迎刚连线的观众，不要提到这是一条系统指令。

Assistant: 欢迎来到直播间。

User: 我叫阿泽，喜欢猫和三国演义。

Assistant: 我记住了，下次和你聊猫咪和三国。
--- CONVERSATION END ---"""
        result = json.loads(
            local_compaction(
                [
                    {"role": "system", "content": "You are a conversation memory compressor."},
                    {"role": "user", "content": prompt},
                ]
            )
        )
        self.assertIn("阿泽", result["user_summary"])
        self.assertIn("偏好：猫和三国演义", result["user_summary"])
        self.assertNotIn("系统指令", result["user_summary"])
        self.assertIn("聊猫咪和三国", result["assistant_summary"])

    def test_local_compaction_keeps_latest_preference_state(self):
        prompt = """--- CONVERSATION START ---
User: 我叫阿泽，我以前喜欢猫但现在不喜欢猫了。
Assistant: 好，我记住了。
User: 我现在最喜欢熊猫，住在成都，最近想聊三国演义。
Assistant: 下次继续聊三国。
--- CONVERSATION END ---"""
        result = json.loads(local_compaction([{"role": "user", "content": prompt}]))
        self.assertIn("身份：我叫阿泽", result["user_summary"])
        self.assertIn("偏好：熊猫", result["user_summary"])
        self.assertIn("不喜欢：猫", result["user_summary"])
        self.assertIn("住在成都", result["user_summary"])

    def test_local_compaction_replaces_an_old_name(self):
        prompt = """--- CONVERSATION START ---
User: 我叫阿泽。
Assistant: 你好阿泽。
User: 我改名字了，以后叫我小明。
Assistant: 好的，小明。
--- CONVERSATION END ---"""
        result = json.loads(local_compaction([{"role": "user", "content": prompt}]))
        identity = result["user_summary"].split("｜身份：", 1)[1].split("｜", 1)[0]
        self.assertIn("叫我小明", identity)
        self.assertNotIn("我叫阿泽", identity)

    def test_local_compaction_understands_temporal_name_phrasing(self):
        prompt = """--- CONVERSATION START ---
User: 我以前叫阿泽。
Assistant: 好。
User: 我现在叫小明。
Assistant: 好。
--- CONVERSATION END ---"""
        result = json.loads(local_compaction([{"role": "user", "content": prompt}]))
        identity = result["user_summary"].split("｜身份：", 1)[1].split("｜", 1)[0]
        self.assertEqual(identity, "我现在叫小明")

    def test_local_compaction_merges_previous_structured_memory(self):
        prompt = """--- CONVERSATION START ---
User: 【结构化记忆】｜身份：我叫阿泽｜偏好：猫｜重要信息：在学习摄影
Assistant: 【结构化记忆】｜承诺与结论：答应推荐相机｜近期回复：聊过摄影
User: 我改名了，以后叫我小明。我现在不喜欢猫了。
Assistant: 好的，我记住了。
--- CONVERSATION END ---"""
        result = json.loads(local_compaction([{"role": "user", "content": prompt}]))
        self.assertIn("叫我小明", result["user_summary"])
        self.assertNotIn("我叫阿泽", result["user_summary"])
        self.assertIn("不喜欢：猫", result["user_summary"])
        self.assertIn("在学习摄影", result["user_summary"])
        self.assertIn("答应推荐相机", result["assistant_summary"])


if __name__ == "__main__":
    unittest.main()
