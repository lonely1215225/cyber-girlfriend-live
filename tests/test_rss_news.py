import sys
import unittest
from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "apps" / "web"
sys.path.insert(0, str(FRONTEND_DIR))

from rss_news import (  # noqa: E402
    IdleNewsRotator,
    RssNewsAggregator,
    formatted_news_blocks,
    infer_topic_filters,
    canonical_news_url,
    news_titles_similar,
    normalize_news_title,
    parse_feed,
)


class RssNewsTests(unittest.TestCase):
    def test_parses_rss_with_attribution_and_time(self):
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel><item>
          <title>Bitcoin rises after market update</title>
          <link>https://example.com/story</link>
          <description><![CDATA[<p>A concise &amp; useful summary.</p>]]></description>
          <source>Example News</source>
          <pubDate>Thu, 20 Aug 2026 08:00:00 GMT</pubDate>
        </item></channel></rss>"""
        items = parse_feed(payload, "Fallback", query_match=True)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source, "Example News")
        self.assertEqual(items[0].summary, "A concise & useful summary.")
        self.assertEqual(items[0].published.year, 2026)
        self.assertTrue(items[0].query_match)

    def test_parses_atom_link_attribute(self):
        payload = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
          <title>World headline</title><link href="https://example.com/atom" />
          <summary>Summary</summary><updated>2026-08-20T08:00:00Z</updated>
        </entry></feed>"""
        item = parse_feed(payload, "Atom Source")[0]
        self.assertEqual(item.link, "https://example.com/atom")
        self.assertEqual(item.source, "Atom Source")

    def test_rejects_oversized_payload(self):
        with self.assertRaises(ValueError):
            parse_feed(b"x" * 1_500_001, "Too Large")

    def test_idle_rotator_avoids_repeating_for_same_caller(self):
        output = """RSS news:
1. [Source A] 2026-08-20 09:00 UTC — First headline
   摘要：First summary
   原文：https://example.com/1
2. [Source B] 2026-08-20 08:00 UTC — Second headline
   摘要：Second summary
   原文：https://example.com/2"""
        blocks = formatted_news_blocks(output)
        self.assertEqual(len(blocks), 2)
        self.assertNotIn("原文", blocks[0])
        rotator = IdleNewsRotator()
        first = rotator.choose("caller-a", output)
        second = rotator.choose("caller-a", output)
        other = rotator.choose("caller-b", output)
        self.assertIn("First headline", first)
        self.assertIn("Second headline", second)
        self.assertIn("First headline", other)

    def test_idle_rotator_cycles_news_technology_and_knowledge(self):
        output = """RSS topics:
1. [新闻｜人民日报] 2026-08-21 09:00 UTC — News headline
2. [科技｜IT之家] 2026-08-21 08:00 UTC — Technology headline
3. [知识｜知乎日报] 2026-08-21 07:00 UTC — Knowledge headline
4. [新闻｜澎湃新闻] 2026-08-21 06:00 UTC — Another news headline"""
        rotator = IdleNewsRotator()
        self.assertIn("News headline", rotator.choose("live-room", output))
        self.assertIn("Technology headline", rotator.choose("live-room", output))
        self.assertIn("Knowledge headline", rotator.choose("live-room", output))
        self.assertIn("Another news headline", rotator.choose("live-room", output))

    def test_parse_feed_keeps_topic_category(self):
        payload = b"""<rss><channel><item><title>Tech story</title>
        <link>https://example.com/tech</link></item></channel></rss>"""
        item = parse_feed(payload, "Tech Source", category="科技")[0]
        self.assertEqual(item.category, "科技")

    def test_infers_requested_category_and_source(self):
        self.assertEqual(infer_topic_filters("讲讲今天的科技新闻"), ("科技", ""))
        self.assertEqual(
            infer_topic_filters("人民日报今天有什么内容"),
            ("新闻", "人民日报"),
        )
        self.assertEqual(
            infer_topic_filters("知乎日报有什么值得聊的"),
            ("知识", "知乎日报"),
        )
        self.assertEqual(infer_topic_filters("今天有什么体育新闻"), ("新闻", ""))

    def test_normalizes_numbers_urls_and_syndicated_titles(self):
        self.assertEqual(normalize_news_title("快讯：比特币突破 8 万美元"), "比特币突破80000美元")
        self.assertEqual(
            canonical_news_url("https://EXAMPLE.com/story/?utm_source=x&id=2#top"),
            "https://example.com/story?id=2",
        )
        self.assertTrue(news_titles_similar("比特币突破8万美元", "比特币价格站上80000美元"))
        self.assertFalse(news_titles_similar("某公司发布新手机", "某公司季度利润大涨"))

    def test_persisted_history_blocks_cross_restart_repeat(self):
        output = """RSS topics:
1. [新闻｜人民日报] 2026-08-21 09:00 UTC — 比特币价格站上80000美元
2. [科技｜IT之家] 2026-08-21 08:00 UTC — 新款芯片正式发布"""
        rotator = IdleNewsRotator()
        selected = rotator.choose(
            "fresh-process", output, persisted_titles=["比特币突破8万美元"]
        )
        self.assertIn("新款芯片", selected)


class RssDialogueQueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_filters_dialogue_query_by_category_and_source(self):
        aggregator = RssNewsAggregator()

        async def latest_topics():
            return """RSS topics:
1. [新闻｜人民日报] 2026-08-21 09:00 UTC — News headline
2. [科技｜IT之家] 2026-08-21 08:00 UTC — Technology headline
3. [知识｜知乎日报] 2026-08-21 07:00 UTC — Knowledge headline"""

        aggregator.latest_topics = latest_topics
        technology = await aggregator.query_topics(category="科技", query="科技新闻")
        self.assertIn("Technology headline", technology)
        self.assertNotIn("News headline", technology)
        zhihu = await aggregator.query_topics(source="知乎日报", query="知乎日报")
        self.assertIn("Knowledge headline", zhihu)
        self.assertNotIn("Technology headline", zhihu)

    async def test_prioritizes_matching_subtopic_inside_category(self):
        aggregator = RssNewsAggregator()

        async def latest_topics():
            return """RSS topics:
1. [新闻｜人民日报] 2026-08-21 09:00 UTC — General headline
2. [新闻｜澎湃新闻] 2026-08-21 08:00 UTC — 体育赛事最新赛果"""

        aggregator.latest_topics = latest_topics
        output = await aggregator.query_topics(query="今天有什么体育新闻", limit=2)
        self.assertLess(output.index("体育赛事"), output.index("General headline"))


if __name__ == "__main__":
    unittest.main()
