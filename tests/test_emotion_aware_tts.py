from __future__ import annotations

import unittest

from tests.import_paths import add_service_paths

add_service_paths()

from emotion_aware_tts import (  # noqa: E402
    choose_tts_style,
    is_leading_interjection,
    prepare_tts_text,
    remove_unspeechable_preserving_cjk,
    split_streaming_sentences,
)
from indextts_client import live_tts_options  # noqa: E402
from playback_policy import (  # noqa: E402
    apply_websocket_playback_policy,
    begin_live_tts_turn,
    is_batch_tts,
    response_is_progress_only,
    set_batch_tts,
    should_complete_flush_before_play,
)
from fish_s2_tags import apply_fish_performance_tags, clean_public_fish_text  # noqa: E402
from sensevoice_stt import SenseVoiceMetadata  # noqa: E402
from expression_director import (  # noqa: E402
    DELIVERY_CONTROL_PROMPT,
    DeliveryControlFilter,
    begin_delivery_generation,
    begin_delivery_response,
    clear_delivery_state,
    cue_duration_ms,
    cues_after,
    publish_expression,
)


class EmotionAwareTTSTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_delivery_state()
        begin_delivery_generation()
        set_batch_tts(False)

    def test_chinese_streaming_split_emits_completed_sentence_immediately(self) -> None:
        first = "三丰，小米现在最新旗舰是小米十七系列。"
        self.assertEqual(split_streaming_sentences(first + "你今天"), [first, "你今天"])
        self.assertEqual(split_streaming_sentences(first), [first, ""])

    def test_short_first_sentence_waits_for_the_next_clause(self) -> None:
        self.assertEqual(split_streaming_sentences("没干嘛。"), ["没干嘛。"])
        self.assertEqual(split_streaming_sentences("你好呀。你今天"), ["你好呀。你今天"])
        self.assertEqual(
            split_streaming_sentences("没干嘛。就……等人，比如现在就在等你说话呢。"),
            ["没干嘛。就……等人，比如现在就在等你说话呢。", ""],
        )

    def test_vocative_and_comma_stay_inside_the_first_sentence(self) -> None:
        self.assertEqual(
            split_streaming_sentences("三丰，小米现在最新旗舰是小米十七系列。后面还有"),
            ["三丰，小米现在最新旗舰是小米十七系列。", "后面还有"],
        )
        self.assertEqual(
            split_streaming_sentences("价格是1,280.50元，今天有效"),
            ["价格是1,280.50元，今天有效"],
        )

    def test_leading_hey_ai_you_are_not_their_own_tts_clip(self) -> None:
        self.assertTrue(is_leading_interjection("嘿，"))
        self.assertTrue(is_leading_interjection("[playful]哎，"))
        self.assertTrue(is_leading_interjection("呦……"))
        self.assertTrue(is_leading_interjection("嘿嘿。"))
        self.assertFalse(is_leading_interjection("小麻呗。"))
        self.assertEqual(
            split_streaming_sentences("哎呦喂……你还真来了。"),
            ["哎呦喂……你还真来了。"],
        )
        self.assertEqual(
            split_streaming_sentences("喂，你先听我说。"),
            ["喂，你先听我说。"],
        )
        self.assertEqual(
            split_streaming_sentences("嘿，我在这儿。下一句"),
            ["嘿，我在这儿。下一句"],
        )
        self.assertEqual(
            split_streaming_sentences("嘿嘿。叫我小麻就行。"),
            ["嘿嘿。叫我小麻就行。"],
        )
        self.assertEqual(prepare_tts_text("喂，", "cheerful"), "喂，")

    def test_explicit_complete_audio_still_keeps_one_tts_request(self) -> None:
        report = "嗯……刚看到新闻，中国人保上半年科技金融投了挺多。"
        streamed = split_streaming_sentences(report)
        self.assertEqual(streamed, [report, ""])
        apply_websocket_playback_policy(complete_audio=True, playback_mode="proactive")
        self.assertTrue(is_batch_tts())
        self.assertEqual(split_streaming_sentences(report), [report])
        apply_websocket_playback_policy(complete_audio=False, playback_mode="interactive")
        self.assertFalse(is_batch_tts())
        self.assertEqual(split_streaming_sentences(report), streamed)
        self.assertFalse(should_complete_flush_before_play("interactive"))
        self.assertTrue(should_complete_flush_before_play("proactive"))
        self.assertTrue(response_is_progress_only(
            type("Resp", (), {"metadata": {"client_purpose": "tool_progress"}})()
        ))
        self.assertFalse(response_is_progress_only(
            type("Resp", (), {"metadata": {"client_purpose": "answer"}})()
        ))
        begin_live_tts_turn()
        self.assertEqual(
            live_tts_options(False),
            {"live": True, "followup": False},
        )
        self.assertEqual(
            live_tts_options(False),
            {"live": True, "followup": True},
        )
        begin_live_tts_turn()
        self.assertEqual(
            live_tts_options(True),
            {"live": False, "followup": False},
        )

    def test_quoted_question_does_not_cut_a_joke_in_half(self) -> None:
        joke = (
            "听好了啊：有一天小明去拔牙，牙医说“别怕”，"
            "小明说“那拔一半行吗？”牙医：“也行。”结果——半拔半留，笑死我了！"
        )
        self.assertEqual(split_streaming_sentences(joke), [joke, ""])
        self.assertEqual(
            split_streaming_sentences("还有啊？行吧行吧，那我可就不客气了。后面"),
            ["还有啊？行吧行吧，那我可就不客气了。", "后面"],
        )
        self.assertEqual(split_streaming_sentences("还有啊？"), ["还有啊？"])

    def test_comma_does_not_start_tts_before_the_sentence_ends(self) -> None:
        self.assertEqual(
            split_streaming_sentences("这条消息最重要的变化是，后面还有更多证据"),
            ["这条消息最重要的变化是，后面还有更多证据"],
        )

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
            '<e profile="laugh" intensity=0.68 style=cheerful mouth=0.16>'
            '见到你当然开心呀。</e>'
        )
        self.assertEqual(visible, "见到你当然开心呀。")
        cue = publish_expression(visible)
        self.assertEqual(cue.source, "llm")
        self.assertEqual(cue.profile, "laugh")
        self.assertAlmostEqual(cue.intensity, 0.68)

    def test_bare_control_line_never_leaks_to_chat_or_tts(self) -> None:
        control = DeliveryControlFilter()
        chunks = ["wi", "nk 0.75 cheer", "ful\n\n欢迎你呀。"]
        visible = "".join(control.feed(chunk) for chunk in chunks)
        self.assertEqual(visible, "欢迎你呀。")
        cue = publish_expression(visible)
        self.assertEqual(cue.profile, "wink")
        self.assertEqual(cue.style, "cheerful")

    def test_textual_tool_call_never_reaches_tts(self) -> None:
        visible = remove_unspeechable_preserving_cjk(
            "<tool_call>web_fetch?url=https://example.com</tool_call>"
        )
        self.assertEqual(visible, "")

    def test_multiple_bare_controls_are_removed_from_any_text_position(self) -> None:
        control = DeliveryControlFilter()
        chunks = [
            "smirk 0.6 cheer", "ful\n哟，敢不敢比？ ha",
            "ppy 0.75 gentle\n输了可别跑。",
        ]
        visible = "".join(control.feed(chunk) for chunk in chunks)
        self.assertEqual(visible, "哟，敢不敢比？ 输了可别跑。")
        first = publish_expression(visible)
        timeline = cues_after(first.sequence - 1)
        self.assertEqual([cue.profile for cue in timeline], ["smirk", "neutral"])

    def test_inline_controls_advance_each_spoken_segment(self) -> None:
        control = DeliveryControlFilter()
        first = control.feed("<e shy 0.62 gentle>刚看到你，我还有点害羞。")
        second = control.feed("<e smirk 0.74 cheerful>不过你别得意得太早呀。")
        first_cue = publish_expression(first + second)
        timeline = cues_after(first_cue.sequence - 1)
        self.assertEqual([cue.profile for cue in timeline], ["shy", "smirk"])
        self.assertEqual(timeline[0].delay_ms, 0)
        self.assertGreater(timeline[1].delay_ms, 0)

    def test_later_tts_segment_keeps_absolute_response_timing(self) -> None:
        control = DeliveryControlFilter()
        first_text = control.feed("<e laugh 0.65 cheerful>第一句先开心地说完。")
        first_cue = publish_expression(first_text)
        second_text = control.feed("<e shy 0.60 gentle>第二句再慢慢害羞。")
        second_cue = publish_expression(second_text)
        self.assertEqual(first_cue.delay_ms, 0)
        self.assertGreater(second_cue.delay_ms, 0)

    def test_streamed_compact_control_is_not_eaten_as_legacy_fields(self) -> None:
        control = DeliveryControlFilter()
        visible = "".join(
            (
                control.feed("laugh 0.58 neutral 0.08"),
                control.feed(" none 1.00\n先说正事。"),
            )
        )
        self.assertEqual(visible, "先说正事。")
        cue = publish_expression(visible)
        self.assertEqual(cue.source, "llm")
        self.assertEqual(cue.profile, "laugh")
        self.assertEqual(cue.vocal_emotion, "neutral")
        self.assertAlmostEqual(cue.vocal_intensity, 0.08)

    def test_natural_text_starting_with_profile_word_is_preserved(self) -> None:
        control = DeliveryControlFilter()
        visible = control.feed("laugh 一下就好，别太认真。", final=True)
        self.assertEqual(visible, "laugh 一下就好，别太认真。")
        cue = publish_expression(visible)
        self.assertEqual(cue.source, "fallback")

    def test_retired_faces_are_remapped_away_from_dropped_portraits(self) -> None:
        control = DeliveryControlFilter()
        visible = control.feed("<e happy 0.68 cheerful>见到你真好。")
        self.assertEqual(visible, "见到你真好。")
        happy_cue = publish_expression(visible)
        self.assertEqual(happy_cue.profile, "neutral")
        self.assertEqual(happy_cue.intensity, 0.0)

        begin_delivery_generation()
        visible = control.feed("<e serious 0.62 serious>先把正事说清楚。")
        self.assertEqual(visible, "先把正事说清楚。")
        serious_cue = publish_expression(visible)
        self.assertEqual(serious_cue.profile, "neutral")
        self.assertEqual(serious_cue.intensity, 0.0)

    def test_new_faces_are_accepted_and_silent_only_holds_after_speech(self) -> None:
        control = DeliveryControlFilter()
        visible = control.feed("<e lip_bite 0.70 playful 0.50 none 1.00>先别急着笑我。")
        self.assertEqual(visible, "先别急着笑我。")
        bite = publish_expression(visible)
        self.assertEqual(bite.profile, "lip_bite")
        self.assertGreaterEqual(bite.duration_ms, 1800)

        begin_delivery_generation()
        visible = control.feed("<e soft_smile 0.55 gentle>见到你就好。")
        smile = publish_expression(visible)
        self.assertEqual(smile.profile, "soft_smile")

    def test_visual_hold_is_based_on_length_not_phrase_matching(self) -> None:
        self.assertEqual(cue_duration_ms("甲乙丙丁", "laugh"), cue_duration_ms("春夏秋冬", "laugh"))

    def test_delivery_prompt_forbids_english_bracket_tags(self) -> None:
        self.assertIn("不要加英文方括号标签", DELIVERY_CONTROL_PROMPT)
        self.assertNotIn("[laughing]", DELIVERY_CONTROL_PROMPT)
        self.assertNotIn("[sigh]", DELIVERY_CONTROL_PROMPT)
        self.assertIn("声音整体必须温柔、软、轻", DELIVERY_CONTROL_PROMPT)
        self.assertIn("优先用 warm 或 tender", DELIVERY_CONTROL_PROMPT)

    def test_fish_tags_survive_speechable_filter_and_are_hidden_from_viewers(self) -> None:
        spoken = "[laughing]嘿嘿，被你发现了。"
        self.assertEqual(remove_unspeechable_preserving_cjk(spoken), spoken)
        self.assertEqual(clean_public_fish_text(spoken), "嘿嘿，被你发现了。")
        self.assertEqual(clean_public_fish_text("先听我说[whis"), "先听我说")

    def test_delivery_plan_injects_fish_tags_without_duplicating_llm_tags(self) -> None:
        self.assertEqual(
            apply_fish_performance_tags(
                "唉，那就这样吧。",
                vocal_emotion="sad",
                vocal_intensity=0.56,
                nonverbal="sigh",
            ),
            "[sigh][sad]唉，那就这样吧。",
        )
        self.assertEqual(
            apply_fish_performance_tags(
                "[laughing]哈哈你别贫。",
                vocal_emotion="happy",
                vocal_intensity=0.6,
                nonverbal="laugh",
            ),
            "[laughing]哈哈你别贫。",
        )
        self.assertEqual(
            apply_fish_performance_tags(
                "今天天气不错。",
                vocal_emotion="happy",
                vocal_intensity=0.1,
                nonverbal="none",
            ),
            "今天天气不错。",
        )


if __name__ == "__main__":
    unittest.main()
