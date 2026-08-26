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
    split_streaming_sentences,
)
from sensevoice_stt import SenseVoiceMetadata  # noqa: E402
from expression_director import (  # noqa: E402
    DeliveryControlFilter,
    begin_delivery_response,
    clear_delivery_state,
    cue_duration_ms,
    publish_expression,
)


class EmotionAwareTTSTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_delivery_state()

    def test_chinese_streaming_split_emits_completed_sentence_immediately(self) -> None:
        self.assertEqual(split_streaming_sentences("你好呀。你今天"), ["你好呀。", "你今天"])
        self.assertEqual(split_streaming_sentences("你好呀。"), ["你好呀。", ""])

    def test_short_vocative_comma_is_an_early_streaming_boundary(self) -> None:
        self.assertEqual(
            split_streaming_sentences("三丰，小米现在最新旗舰是小米十七系列。后面还有"),
            ["三丰，", "小米现在最新旗舰是小米十七系列。", "后面还有"],
        )
        self.assertEqual(
            split_streaming_sentences("价格是1,280.50元，今天有效"),
            ["价格是1,280.50元，今天有效"],
        )

    def test_short_opening_clause_streams_without_a_phrase_dictionary(self) -> None:
        self.assertEqual(
            split_streaming_sentences("哎呦喂……你还真来了。"),
            ["哎呦喂……", "你还真来了。", ""],
        )
        self.assertEqual(
            split_streaming_sentences("喂，你先听我说。"),
            ["喂，", "你先听我说。", ""],
        )
        self.assertEqual(prepare_tts_text("喂，", "cheerful"), "喂，")

    def test_fallback_style_does_not_guess_semantics_from_words(self) -> None:
        self.assertEqual(choose_tts_style("任意复杂语境", SenseVoiceMetadata(emotion="难过")), "neutral")
        self.assertEqual(choose_tts_style("你今天在做什么？"), "neutral")

    def test_spoken_text_preserves_model_punctuation(self) -> None:
        self.assertEqual(prepare_tts_text("嗯...我在呢", "gentle"), "嗯……我在呢")
        self.assertEqual(prepare_tts_text("欢迎回来！", "cheerful"), "欢迎回来！")
        self.assertEqual(
            prepare_tts_text("航班 CA1234 于 2026 年 8 月 21 日延误 30 分钟。", "neutral"),
            "航班 CA1234 于 2026 年 8 月 21 日延误 30 分钟。",
        )
        self.assertEqual(
            prepare_tts_text("三丰，小米现在最新旗舰是小米十七系列。", "neutral"),
            "三丰，\n小米现在最新旗舰是小米十七系列。",
        )
        self.assertEqual(
            prepare_tts_text("价格是1,280.50元，编号CA1234不能拆。", "neutral"),
            "价格是1,280.50元， 编号CA1234不能拆。",
        )
        self.assertEqual(
            prepare_tts_text(
                "这是一段已经明显超过二十个汉字而且确实需要自然呼吸的内容，后面继续说。",
                "neutral",
            ),
            "这是一段已经明显超过二十个汉字而且确实需要自然呼吸的内容，\n后面继续说。",
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

    def test_llm_control_tag_is_hidden_and_drives_the_next_tts_segment(self) -> None:
        control = DeliveryControlFilter()
        chunks = ["<e sm", "irk 0.76 cheerful 0.18>", "你还挺会说嘛。"]
        visible = "".join(control.feed(chunk) for chunk in chunks)
        self.assertEqual(visible, "你还挺会说嘛。")
        cue = publish_expression(visible)
        self.assertEqual(cue.source, "llm")
        self.assertEqual(cue.profile, "smirk")
        self.assertEqual(cue.style, "cheerful")
        self.assertAlmostEqual(cue.intensity, 0.76)
        next_segment = publish_expression("同一轮的下一句话。")
        self.assertEqual(next_segment.profile, "smirk")
        begin_delivery_response()
        following_response = publish_expression("下一轮没有控制标签。")
        self.assertEqual(following_response.source, "fallback")

    def test_invalid_or_missing_control_never_guesses_from_dialogue_text(self) -> None:
        control = DeliveryControlFilter()
        self.assertEqual(control.feed("<e invented 0.9 loud 0.9>正文"), "正文")
        cue = publish_expression("无论正文表达什么，都不在代码中猜语义。")
        self.assertEqual(cue.source, "fallback")
        self.assertEqual(cue.profile, "neutral")
        self.assertEqual(cue.intensity, 0.0)

    def test_local_model_attribute_serialization_and_closing_tag_are_hidden(self) -> None:
        control = DeliveryControlFilter()
        visible = control.feed(
            '<e profile="happy" intensity=0.68 style=cheerful mouth=0.16>'
            '见到你当然开心呀。</e>'
        )
        self.assertEqual(visible, "见到你当然开心呀。")
        cue = publish_expression(visible)
        self.assertEqual(cue.source, "llm")
        self.assertEqual(cue.profile, "happy")
        self.assertAlmostEqual(cue.intensity, 0.68)

    def test_visual_hold_is_based_on_length_not_phrase_matching(self) -> None:
        self.assertEqual(cue_duration_ms("甲乙丙丁", "happy"), cue_duration_ms("春夏秋冬", "happy"))


if __name__ == "__main__":
    unittest.main()
