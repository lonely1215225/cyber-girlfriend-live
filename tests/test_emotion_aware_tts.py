from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROXY = Path(__file__).resolve().parents[1] / "proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

from emotion_aware_tts import (  # noqa: E402
    choose_tts_style,
    prepare_tts_text,
    remove_unspeechable_preserving_cjk,
)
from sensevoice_stt import SenseVoiceMetadata  # noqa: E402


class EmotionAwareTTSTests(unittest.TestCase):
    def test_user_emotion_has_priority(self) -> None:
        self.assertEqual(
            choose_tts_style("欢迎回来", SenseVoiceMetadata(emotion="难过")),
            "gentle",
        )
        self.assertEqual(
            choose_tts_style("没事的", SenseVoiceMetadata(emotion="开心")),
            "cheerful",
        )
        self.assertEqual(
            choose_tts_style("我理解你", SenseVoiceMetadata(emotion="生气")),
            "calm",
        )

    def test_reply_semantics_are_used_without_audio_context(self) -> None:
        self.assertEqual(choose_tts_style("别怕，我在这里陪你。"), "gentle")
        self.assertEqual(choose_tts_style("嘿，欢迎来到直播间！"), "cheerful")
        self.assertEqual(choose_tts_style("目前这条新闻仍需确认。"), "serious")
        self.assertEqual(choose_tts_style("你今天在做什么？"), "neutral")

    def test_spoken_text_preserves_model_punctuation(self) -> None:
        self.assertEqual(prepare_tts_text("嗯...我在呢", "gentle"), "嗯……我在呢")
        self.assertEqual(prepare_tts_text("欢迎回来！", "cheerful"), "欢迎回来！")
        self.assertEqual(
            prepare_tts_text("航班 CA1234 于 2026 年 8 月 21 日延误 30 分钟。", "neutral"),
            "航班 CA1234 于 2026 年 8 月 21 日延误 30 分钟。",
        )

    def test_does_not_invent_punctuation_for_unpunctuated_output(self) -> None:
        self.assertEqual(
            prepare_tts_text("飞机编号B12345前后保持连续", "neutral"),
            "飞机编号B12345前后保持连续",
        )

    def test_cjk_punctuation_survives_speechable_filter(self) -> None:
        original = "航班CA1234，于2026年8月21日延误30分钟；请留意！价格是￥1,280.50。"
        self.assertEqual(
            remove_unspeechable_preserving_cjk(original + "🙂"),
            original,
        )


if __name__ == "__main__":
    unittest.main()
