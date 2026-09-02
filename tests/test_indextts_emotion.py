from __future__ import annotations

import unittest

from tests.import_paths import add_service_paths

add_service_paths()

from indextts_emotion import (  # noqa: E402
    INDEX_EMOTION_ORDER,
    clamp_duration_factor,
    emo_alpha_from_intensity,
    emo_vector_for,
    plan_indextts_controls,
    soften_vector,
    strip_english_bracket_tags,
)


class IndexTTSEmotionTests(unittest.TestCase):
    def test_vector_order_and_neutral_is_calm(self) -> None:
        self.assertEqual(len(INDEX_EMOTION_ORDER), 8)
        vector = emo_vector_for("neutral")
        self.assertEqual(len(vector), 8)
        self.assertGreater(vector[INDEX_EMOTION_ORDER.index("calm")], 0.8)
        self.assertLess(vector[INDEX_EMOTION_ORDER.index("happy")], 0.08)

    def test_playful_is_not_full_happy(self) -> None:
        playful = emo_vector_for("playful")
        happy = emo_vector_for("happy")
        self.assertLess(
            playful[INDEX_EMOTION_ORDER.index("happy")],
            happy[INDEX_EMOTION_ORDER.index("happy")],
        )
        self.assertGreater(playful[INDEX_EMOTION_ORDER.index("calm")], 0.5)

    def test_daily_happy_is_pulled_toward_tender(self) -> None:
        raw = emo_vector_for("happy")
        soft = soften_vector("happy", 0.08)
        self.assertLess(
            soft[INDEX_EMOTION_ORDER.index("happy")],
            raw[INDEX_EMOTION_ORDER.index("happy")],
        )
        self.assertGreater(
            soft[INDEX_EMOTION_ORDER.index("calm")],
            raw[INDEX_EMOTION_ORDER.index("calm")],
        )
        self.assertGreater(
            plan_indextts_controls(vocal_emotion="happy", vocal_intensity=0.08)[
                "emo_vector"
            ][INDEX_EMOTION_ORDER.index("melancholic")],
            0.15,
        )

    def test_marked_tease_keeps_playful_color(self) -> None:
        raw = emo_vector_for("playful")
        tease = plan_indextts_controls(
            vocal_emotion="playful",
            vocal_intensity=0.40,
            nonverbal="none",
            spoken_text="你还挺会说嘛。",
        )
        self.assertEqual(tease["emo_vector"], raw)
        self.assertTrue(tease["use_emo_text"])
        self.assertIn("坏笑", tease["emo_text"])
        self.assertGreaterEqual(tease["emo_alpha"], 0.24)

    def test_alpha_stays_soft(self) -> None:
        self.assertAlmostEqual(emo_alpha_from_intensity(0.0), 0.10)
        self.assertLessEqual(emo_alpha_from_intensity(0.18), 0.16)
        self.assertLessEqual(emo_alpha_from_intensity(0.5), 0.32)
        self.assertLessEqual(emo_alpha_from_intensity(1.0), 0.40)
        self.assertGreaterEqual(emo_alpha_from_intensity(0.40, "playful"), 0.24)
        self.assertLessEqual(emo_alpha_from_intensity(1.0, "happy"), 0.50)

    def test_text_gate_is_closed_for_quiet_news(self) -> None:
        plan = plan_indextts_controls(
            vocal_emotion="neutral",
            vocal_intensity=0.08,
            nonverbal="none",
            spoken_text="刚看到新闻，中国人保上半年科技金融投了挺多。",
        )
        self.assertFalse(plan["use_emo_text"])
        self.assertEqual(plan["emo_text"], "")
        self.assertIsNone(plan["emo_audio"])

    def test_high_intensity_or_laugh_opens_text(self) -> None:
        tease = plan_indextts_controls(
            vocal_emotion="playful",
            vocal_intensity=0.52,
            nonverbal="none",
            spoken_text="你还挺会说嘛。",
        )
        self.assertTrue(tease["use_emo_text"])
        self.assertIn("坏笑", tease["emo_text"])
        laugh = plan_indextts_controls(
            vocal_emotion="happy",
            vocal_intensity=0.1,
            nonverbal="laugh",
            spoken_text="[laughing]嘿嘿，被你发现了。",
        )
        self.assertTrue(laugh["use_emo_text"])
        self.assertIn("笑", laugh["emo_text"])
        self.assertEqual(laugh["spoken_text"], "嘿嘿，被你发现了。")

    def test_duration_factor_is_clamped(self) -> None:
        self.assertEqual(clamp_duration_factor(1.4), 1.06)
        self.assertEqual(clamp_duration_factor(0.5), 1.00)
        self.assertEqual(plan_indextts_controls(duration_factor=1.03)["duration_factor"], 1.03)

    def test_english_bracket_tags_are_stripped(self) -> None:
        self.assertEqual(strip_english_bracket_tags("[sigh]唉……那我慢慢说。"), "唉……那我慢慢说。")


if __name__ == "__main__":
    unittest.main()
