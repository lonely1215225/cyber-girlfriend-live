import asyncio
import sys
import unittest
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "apps" / "web"
sys.path.insert(0, str(FRONTEND_DIR))

from dialogue_intent import LOOKUP_FAIL_LINE, SIMPLE_CHAT_POLICY  # noqa: E402
from voice_lookup import (  # noqa: E402
    VoiceLookupGate,
    compose_voice_instructions,
    evidence_turn_messages,
    run_voice_lookup,
    strip_live_tools,
)


class VoiceLookupTests(unittest.IsolatedAsyncioTestCase):
    def test_voice_prompt_does_not_ask_the_model_to_search(self):
        text = compose_voice_instructions("你叫小麻。", "张三丰")
        self.assertIn("张三丰", text)
        self.assertIn("简单闲聊", text)
        self.assertIn(SIMPLE_CHAT_POLICY[:8], text)
        self.assertNotIn("smart_web_search", text)
        self.assertNotIn("调用", text)

    def test_evidence_prompt_uses_looked_up_policy(self):
        text = compose_voice_instructions("你叫小麻。", "张三丰", evidence=True)
        self.assertIn("已经查到", text)
        self.assertNotIn("smart_web_search", text)

    def test_live_session_strips_tools(self):
        session = {"type": "realtime", "tools": [{"name": "smart_web_search"}]}
        strip_live_tools(session)
        self.assertEqual(session["tools"], [])
        self.assertEqual(session["tool_choice"], "none")

    def test_gate_drops_auto_response_until_own_turn(self):
        gate = VoiceLookupGate()
        gate.begin("看看有啥新闻不")
        self.assertTrue(gate.should_drop_upstream("response.created"))
        self.assertTrue(gate.should_drop_upstream("response.audio.delta"))
        self.assertFalse(gate.should_drop_upstream("conversation.item.input_audio_transcription.completed"))
        gate.allow_own_responses()
        self.assertFalse(gate.should_drop_upstream("response.created"))

    def test_evidence_item_carries_prefetch_and_utterance(self):
        messages = evidence_turn_messages(
            "instructions", "1. 某国发布了新政策", "张三丰", "看看有啥新闻不"
        )
        self.assertEqual(messages[0]["session"]["tools"], [])
        user_text = messages[1]["item"]["content"][0]["text"]
        self.assertIn("【已查到的资料】", user_text)
        self.assertIn("某国发布了新政策", user_text)
        self.assertIn("看看有啥新闻不", user_text)

    async def test_prefetch_injects_evidence_and_cancels_auto_answer(self):
        sent: list[dict] = []

        async def send(payload):
            sent.append(payload)

        async def prefetch(_query):
            return "刚才查到的资料：\n1. 某国发布了新政策"

        async def pin_news(_query):
            raise AssertionError("news ask should prefetch, not pin")

        gate = VoiceLookupGate()
        gate.begin("看看有啥新闻不")
        await run_voice_lookup(
            utterance="看看有啥新闻不",
            display_name="张三丰",
            persona_prompt="你叫小麻。",
            personal_memory="",
            companion=False,
            kind="prefetch",
            send=send,
            prefetch=prefetch,
            pin_news=pin_news,
            gate=gate,
        )
        self.assertEqual(sent[0]["type"], "response.cancel")
        user_text = next(
            item["item"]["content"][0]["text"]
            for item in sent
            if item.get("type") == "conversation.item.create"
        )
        self.assertIn("【已查到的资料】", user_text)
        self.assertIn("某国发布了新政策", user_text)
        creates = [item for item in sent if item.get("type") == "response.create"]
        self.assertTrue(creates)
        self.assertNotEqual(
            creates[-1].get("response", {}).get("metadata", {}).get("client_purpose"),
            "tool_progress",
        )
        self.assertFalse(gate.holding)

    async def test_failed_prefetch_speaks_fail_line(self):
        sent: list[dict] = []

        async def send(payload):
            sent.append(payload)

        async def prefetch(_query):
            raise RuntimeError("tavily: ConnectTimeout")

        gate = VoiceLookupGate()
        gate.begin("看看最新新闻")
        await run_voice_lookup(
            utterance="看看最新新闻",
            display_name="张三丰",
            persona_prompt="你叫小麻。",
            personal_memory="",
            companion=False,
            kind="prefetch",
            send=send,
            prefetch=prefetch,
            pin_news=lambda _q: asyncio.sleep(0, result=""),
            gate=gate,
        )
        spoken = [
            item["item"]["content"][0]["text"]
            for item in sent
            if item.get("type") == "conversation.item.create"
        ]
        self.assertTrue(any(LOOKUP_FAIL_LINE in text for text in spoken))
        self.assertFalse(any("【已查到的资料】" in text for text in spoken))

    async def test_slow_prefetch_speaks_wait_line_first(self):
        sent: list[dict] = []

        async def send(payload):
            sent.append(payload)
            if payload.get("type") == "response.create":
                gate.observe({
                    "type": "response.done",
                    "response": {"metadata": {"client_purpose": "tool_progress"}},
                })

        async def prefetch(_query):
            await asyncio.sleep(0.05)
            return "刚才查到的资料：\n1. 某国发布了新政策"

        gate = VoiceLookupGate()
        gate.begin("看看最新新闻")
        await run_voice_lookup(
            utterance="看看最新新闻",
            display_name="张三丰",
            persona_prompt="你叫小麻。",
            personal_memory="",
            companion=False,
            kind="prefetch",
            send=send,
            prefetch=prefetch,
            pin_news=lambda _q: asyncio.sleep(0, result=""),
            gate=gate,
        )
        texts = [
            item["item"]["content"][0]["text"]
            for item in sent
            if item.get("type") == "conversation.item.create"
        ]
        self.assertTrue(any("逐字朗读：我翻一下今天的，马上说。" in text for text in texts))
        self.assertTrue(any("【已查到的资料】" in text for text in texts))

    async def test_pin_news_uses_room_topic_without_search(self):
        sent: list[dict] = []
        queries: list[str] = []

        async def send(payload):
            sent.append(payload)

        async def prefetch(_query):
            raise AssertionError("follow-up must not hit the search API")

        async def pin_news(query):
            queries.append(query)
            return "刚才播过：某国发布了新政策"

        gate = VoiceLookupGate()
        gate.begin("刚才那条怎么样")
        await run_voice_lookup(
            utterance="刚才那条怎么样",
            display_name="张三丰",
            persona_prompt="你叫小麻。",
            personal_memory="",
            companion=False,
            kind="pin_news",
            send=send,
            prefetch=prefetch,
            pin_news=pin_news,
            gate=gate,
        )
        self.assertEqual(queries, ["刚才那条怎么样"])
        user_text = next(
            item["item"]["content"][0]["text"]
            for item in sent
            if item.get("type") == "conversation.item.create"
        )
        self.assertIn("【刚才播过的话题】", user_text)
        self.assertIn("某国发布了新政策", user_text)


if __name__ == "__main__":
    unittest.main()
