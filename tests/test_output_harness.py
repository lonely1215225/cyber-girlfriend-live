from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROXY = Path(__file__).resolve().parents[1] / "proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

from output_harness import PublicOutputFilter, clean_public_output  # noqa: E402


class PublicOutputHarnessTests(unittest.TestCase):
    def test_html_breaks_are_removed_but_text_is_preserved(self) -> None:
        self.assertEqual(
            clean_public_output("第一句<br>第二句<p>第三句</p>"),
            "第一句\n第二句\n第三句",
        )

    def test_html_tag_split_across_stream_chunks_never_leaks(self) -> None:
        cleaner = PublicOutputFilter()
        visible = "".join(
            (
                cleaner.feed("你好<b"),
                cleaner.feed("r />三丰<strong>"),
                cleaner.feed("来啦</strong>", final=True),
            )
        )
        self.assertEqual(visible, "你好\n三丰来啦")

    def test_delivery_control_tag_remains_available_to_expression_filter(self) -> None:
        self.assertEqual(
            clean_public_output("<e happy 0.7 cheerful>见到你真好。"),
            "<e happy 0.7 cheerful>见到你真好。",
        )


if __name__ == "__main__":
    unittest.main()
