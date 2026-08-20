import sys
import unittest
from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "s2s" / "hf-realtime-voice"
sys.path.insert(0, str(FRONTEND_DIR))

from rss_news import IdleNewsRotator, formatted_news_blocks, parse_feed  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
