from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROXY = Path(__file__).resolve().parents[1] / "proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

from emotion_aware_tts import choose_tts_style, prepare_tts_text  # noqa: E402
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

    def test_spoken_text_adds_pauses_without_changing_words(self) -> None:
        self.assertEqual(prepare_tts_text("嗯...我在呢", "gentle"), "嗯……我在呢。")
        self.assertEqual(prepare_tts_text("欢迎回来", "cheerful"), "欢迎回来！")
        self.assertEqual(prepare_tts_text("这是正常回答", "neutral"), "这是正常回答。")

    def test_restores_semantic_pauses_for_weak_llm_output(self) -> None:
        self.assertEqual(
            prepare_tts_text(
                "确实挺刺激的不过这正是冒险的魅力所在你最喜欢看哪种惊险的环节呀",
                "neutral",
            ),
            "确实挺刺激的，不过这正是冒险的魅力所在。你最喜欢看哪种惊险的环节呀？",
        )
        self.assertEqual(
            prepare_tts_text(
                "刚下播在整理大家的弹幕你呢是不是也刚刷完剧来逛逛", "neutral"
            ),
            "刚下播在整理大家的弹幕。你呢，是不是也刚刷完剧来逛逛？",
        )

    def test_empathy_pivots_get_a_soft_breath(self) -> None:
        self.assertEqual(
            prepare_tts_text("嗯我在呢别怕我会一直陪着你的", "gentle"),
            "嗯，我在呢，别怕我会一直陪着你的。",
        )

    def test_question_intonation_is_inferred(self) -> None:
        self.assertEqual(
            prepare_tts_text("你今天在做什么", "neutral"), "你今天在做什么？"
        )
        self.assertEqual(prepare_tts_text("嗯我在呢", "gentle"), "嗯，我在呢。")


if __name__ == "__main__":
    unittest.main()
