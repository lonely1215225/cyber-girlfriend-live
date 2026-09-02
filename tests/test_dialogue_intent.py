import sys
import unittest
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "apps" / "web"
sys.path.insert(0, str(FRONTEND_DIR))

from dialogue_intent import (  # noqa: E402
    is_news_request,
    lookup_wait_line,
    looks_like_english_answer,
    looks_like_search_filler,
    needs_web_search,
    viewer_utterance,
    wants_news_context,
)


class DialogueIntentTests(unittest.TestCase):
    def test_casual_chat_is_not_a_live_lookup(self):
        for text in ("哦", "什么玩意", "别说别的了，你就简单讲个笑话", "想我没？"):
            self.assertFalse(needs_web_search(text), text)
            self.assertFalse(wants_news_context(text), text)

    def test_explicit_lookup_needs_search(self):
        self.assertTrue(needs_web_search("帮我查一下现在比特币多少钱"))
        self.assertTrue(needs_web_search("小米最新的汽车是什么，多少钱"))
        self.assertTrue(needs_web_search("今天有什么新闻"))
        self.assertTrue(needs_web_search("你查一下今天有啥好玩的新闻不"))
        self.assertTrue(needs_web_search("看看最新新闻"))
        self.assertTrue(needs_web_search("看看新闻"))
        self.assertTrue(is_news_request("看看最新新闻"))
        self.assertTrue(is_news_request("@小麻 看看最新新闻"))
        self.assertFalse(is_news_request("别说新闻了，你就简单讲个笑话"))
        self.assertFalse(is_news_request("帮我查一下现在比特币多少钱"))

    def test_search_filler_is_not_an_answer(self):
        self.assertTrue(looks_like_search_filler("我正在联网查找相关资料，再核对一下来源呀。"))
        self.assertFalse(looks_like_search_filler("今天有条挺好玩的，一只海豹上了路。"))

    def test_lookup_wait_line_is_a_short_spoken_beat(self):
        self.assertEqual(lookup_wait_line("看看最新新闻"), "我翻一下今天的，马上说。")
        self.assertEqual(lookup_wait_line("帮我查一下现在比特币多少钱"), "我去对一下最新的数。")
        self.assertEqual(lookup_wait_line("查证一下"), "我去看一眼，马上回你。")

    def test_packed_memory_does_not_steal_the_current_line(self):
        packed = (
            "【历史记忆，仅供理解】\n数字人：第一块瓜是今天最新新闻\n\n"
            "【当前评论，这是唯一需要回答的问题】\n"
            "直播间观众“张三丰”说：你就简单讲个笑话"
        )
        self.assertEqual(viewer_utterance(packed), "你就简单讲个笑话")
        self.assertFalse(needs_web_search(packed))

    def test_english_news_dump_is_not_a_spoken_answer(self):
        self.assertTrue(looks_like_english_answer(
            "Based on the search results provided, here is a summary of the latest news."
        ))
        self.assertFalse(looks_like_english_answer("今天有条挺好玩的，一只海豹上了路。"))


if __name__ == "__main__":
    unittest.main()
