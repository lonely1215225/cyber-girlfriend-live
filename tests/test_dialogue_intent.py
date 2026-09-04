import sys
import unittest
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "apps" / "web"
sys.path.insert(0, str(FRONTEND_DIR))

from dialogue_intent import (  # noqa: E402
    LOOKUP_FAIL_LINE,
    SIMPLE_CHAT_POLICY,
    decide_voice_turn,
    is_news_request,
    lookup_fail_line,
    lookup_wait_line,
    looks_like_english_answer,
    looks_like_search_filler,
    needs_web_search,
    news_search_query,
    viewer_utterance,
    wants_news_context,
)


class DialogueIntentTests(unittest.TestCase):
    def test_simple_chat_must_give_the_asked_thing_first(self):
        self.assertIn("第一句就必须点出来", SIMPLE_CHAT_POLICY)
        self.assertIn("禁止只预告、卖关子或反问", SIMPLE_CHAT_POLICY)
        self.assertIn("八十字", SIMPLE_CHAT_POLICY)
        self.assertNotIn("四十个字", SIMPLE_CHAT_POLICY)

    def test_casual_chat_is_not_a_live_lookup(self):
        for text in (
            "哦", "什么玩意", "别说别的了，你就简单讲个笑话", "想我没？",
            "讲个笑话", "西瓜甜不甜", "你咋了", "今天心情不好",
        ):
            self.assertFalse(needs_web_search(text), text)
            self.assertFalse(wants_news_context(text), text)

    def test_explicit_lookup_needs_search(self):
        news_asks = (
            "今天有什么新闻", "你查一下今天有啥好玩的新闻不", "看看最新新闻",
            "看看新闻", "@小麻 看看最新有什么新闻说来听听", "@小麻 看看有啥新闻不",
            "有新闻吗", "说个新闻", "新闻呢", "热搜", "看看热搜", "今日热榜",
            "来点瓜", "吃瓜", "有啥瓜", "今天啥瓜", "网上都在说啥",
            "今天发生了什么", "最近出啥事了", "外面出什么事了",
            "想听今天的资讯", "给我讲讲时事", "刷刷早报",
            "报个新闻", "来段热搜", "整点瓜", "热搜呗",
            "世界怎么了", "今天有啥大事", "刷一下热搜榜",
            "国际上发生了什么", "今日要闻", "听听财经新闻",
        )
        for text in news_asks:
            self.assertTrue(needs_web_search(text), text)
            self.assertTrue(is_news_request(text), text)
        self.assertTrue(needs_web_search("帮我查一下现在比特币多少钱"))
        self.assertTrue(needs_web_search("小米最新的汽车是什么，多少钱"))
        self.assertTrue(needs_web_search("比特币涨了没"))
        self.assertTrue(needs_web_search("今天天气怎么样"))
        self.assertTrue(needs_web_search("会下雨吗"))
        self.assertTrue(needs_web_search("百度一下"))
        self.assertTrue(is_news_request("看看最新新闻"))
        self.assertTrue(is_news_request("@小麻 看看最新新闻"))
        self.assertTrue(is_news_request("@小麻 看看最新有什么新闻说来听听"))
        self.assertTrue(is_news_request("@小麻 看看有啥新闻不"))
        self.assertEqual(
            lookup_wait_line("@小麻 看看最新有什么新闻说来听听"),
            "我翻一下今天的，马上说。",
        )
        self.assertEqual(lookup_wait_line("来点瓜"), "我翻一下今天的，马上说。")
        self.assertEqual(lookup_wait_line("比特币涨了没"), "我去对一下最新的数。")
        self.assertFalse(is_news_request("别说新闻了，你就简单讲个笑话"))
        self.assertFalse(is_news_request("不想听新闻"))
        self.assertFalse(is_news_request("帮我查一下现在比特币多少钱"))
        self.assertFalse(needs_web_search("没有新闻就算了"))
        self.assertFalse(needs_web_search("别说新闻了"))
        self.assertFalse(needs_web_search("不看新闻"))
        self.assertFalse(needs_web_search("别查了"))
        self.assertFalse(needs_web_search("你冷不冷"))
        self.assertTrue(needs_web_search("今天冷不冷"))
        self.assertTrue(needs_web_search("外面要带伞吗"))
        self.assertFalse(wants_news_context("这个好玩吗"))
        self.assertEqual(decide_voice_turn("看看有啥新闻不", tools_enabled=True), "prefetch")
        self.assertEqual(decide_voice_turn("刚才那条怎么样", tools_enabled=True), "pin_news")
        self.assertEqual(decide_voice_turn("今天有点烦", tools_enabled=True), "chat")
        self.assertEqual(decide_voice_turn("看看有啥新闻不", tools_enabled=False), "chat")
        self.assertEqual(lookup_fail_line(), LOOKUP_FAIL_LINE)
        self.assertTrue(wants_news_context("刚才那条怎么样"))
        self.assertTrue(wants_news_context("这个为什么会上涨"))
        self.assertTrue(wants_news_context("辟谣了吗"))
        self.assertEqual(news_search_query("看看有啥新闻不"), "今天国内外热点新闻")
        self.assertEqual(news_search_query("有新闻吗"), "今天国内外热点新闻")
        self.assertEqual(news_search_query("来点瓜"), "今天国内外热点新闻")
        self.assertEqual(news_search_query("看看特斯拉有啥新闻"), "看看特斯拉有啥新闻")
        self.assertEqual(news_search_query("帮我查一下现在比特币多少钱"), "帮我查一下现在比特币多少钱")

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
