import asyncio
import unittest

from tests.import_paths import add_service_paths

add_service_paths()

import avtr1_gateway as gateway  # noqa: E402


class SpeechReservoirTests(unittest.TestCase):
    def setUp(self):
        gateway.speech_finished = False
        gateway.speech_turn_active = False
        gateway.speech_playing = False
        gateway.speech_rebuffering = False
        gateway.speech_dynamic_buffer_bytes = gateway.SPEECH_START_BUFFER_BYTES
        gateway.speech_turn_underruns = 0
        gateway.speech_buffer_underruns = 0
        gateway.speech_silence_inserted_ms = 0
        gateway.speech_turns_completed = 0
        gateway.speech_stable_turns = 0

    def test_waits_for_initial_watermark_without_consuming_audio(self):
        buf = bytearray(b"\x01\x00" * (gateway.SAMPLE_RATE // 2))
        gateway.speech_turn_active = True
        before = bytes(buf)

        _cur, _future, played, has_speech = gateway._window_from_speech(buf)

        self.assertFalse(has_speech)
        self.assertEqual(bytes(buf), before)
        self.assertEqual(set(played), {0})
        self.assertFalse(gateway.speech_rebuffering)

    def test_starts_at_watermark_and_consumes_exactly_200ms(self):
        buf = bytearray(b"\x01\x00" * (gateway.SPEECH_START_BUFFER_BYTES // 2))
        gateway.speech_turn_active = True
        before = len(buf)

        _cur, _future, played, has_speech = gateway._window_from_speech(buf)

        self.assertTrue(has_speech)
        self.assertTrue(gateway.speech_playing)
        self.assertEqual(len(played), gateway.CURRENT_SAMPLES * 2)
        self.assertEqual(before - len(buf), gateway.CURRENT_SAMPLES * 2)

    def test_one_underrun_enters_rebuffering_and_raises_watermark(self):
        buf = bytearray(b"\x01\x00" * (gateway.WINDOW_SAMPLES - 1))
        gateway.speech_turn_active = True
        gateway.speech_playing = True
        before = bytes(buf)

        _cur, _future, _played, has_speech = gateway._window_from_speech(buf)

        self.assertFalse(has_speech)
        self.assertEqual(bytes(buf), before)
        self.assertTrue(gateway.speech_rebuffering)
        self.assertEqual(gateway.speech_buffer_underruns, 1)
        self.assertEqual(
            gateway.speech_dynamic_buffer_bytes,
            gateway.SPEECH_START_BUFFER_BYTES + gateway.SPEECH_REBUFFER_STEP_BYTES,
        )

        # Repeated renderer ticks while waiting must not count as new underruns.
        gateway._window_from_speech(buf)
        self.assertEqual(gateway.speech_buffer_underruns, 1)

        buf.extend(b"\x01\x00" * (gateway.speech_dynamic_buffer_bytes // 2))
        _cur, _future, _played, has_speech = gateway._window_from_speech(buf)
        self.assertTrue(has_speech)
        self.assertFalse(gateway.speech_rebuffering)

    def test_finished_short_turn_drains_without_waiting_for_watermark(self):
        buf = bytearray(b"\x01\x00" * (gateway.SAMPLE_RATE // 10))
        gateway.speech_turn_active = True
        gateway.speech_finished = True

        _cur, _future, played, has_speech = gateway._window_from_speech(buf)

        self.assertTrue(has_speech)
        self.assertEqual(len(played), gateway.CURRENT_SAMPLES * 2)
        self.assertFalse(buf)
        self.assertFalse(gateway.speech_turn_active)
        self.assertEqual(gateway.speech_turns_completed, 1)


class FixedFrameEncoderTests(unittest.TestCase):
    def test_same_raw_frame_can_be_encoded_at_successive_pts(self):
        width, height = 64, 64
        encoder = gateway.H264Encoder(width, height)
        raw = bytes(width * height * 3 // 2)

        first = encoder.encode(raw)
        held = encoder.encode(raw)

        self.assertTrue(first)
        self.assertTrue(held)
        self.assertEqual(encoder.pts, 2)


class AuthoritativeAudioClockTests(unittest.TestCase):
    def setUp(self):
        gateway.clear_av_frames()
        gateway.speech_pcm.clear()
        gateway.speech_output_pcm.clear()
        gateway.speech_turn_active = False
        gateway.speech_playing = False
        gateway.speech_finished = False
        gateway.speech_output_active = False
        gateway.speech_output_finished = False
        gateway.speech_output_ready = False
        gateway.speech_output_rebuffering = False
        gateway.speech_output_target_frames = gateway.OUTPUT_AV_TARGET_FRAMES
        gateway.audio_output_underruns = 0

    def test_pcm_continues_even_when_no_rendered_video_frame_exists(self):
        pcm = bytes(range(256)) * 5
        self.assertEqual(len(pcm), 2 * gateway.PCM_PACKET_BYTES)
        gateway.speech_output_pcm.extend(pcm)
        gateway.speech_output_active = True
        gateway.speech_output_ready = True

        chunks, had_speech = gateway._take_output_audio()

        self.assertTrue(had_speech)
        self.assertEqual(b"".join(chunks), pcm)
        self.assertFalse(gateway.speech_output_pcm)
        self.assertEqual(gateway.audio_output_underruns, 0)

    def test_real_pcm_starvation_reenters_buffering_only_once(self):
        gateway.speech_output_active = True
        gateway.speech_output_ready = True

        chunks, had_speech = gateway._take_output_audio()

        self.assertFalse(had_speech)
        self.assertEqual(set(b"".join(chunks)), {0})
        self.assertFalse(gateway.speech_output_ready)
        self.assertTrue(gateway.speech_output_rebuffering)
        self.assertEqual(gateway.audio_output_underruns, 1)
        gateway._take_output_audio()
        self.assertEqual(gateway.audio_output_underruns, 1)

    def test_proactive_turn_uses_deeper_video_reservoir_and_copies_pcm(self):
        pcm = b"\x01\x00" * gateway.SAMPLE_RATE

        asyncio.run(gateway.append_speech(pcm, mode="proactive"))

        self.assertEqual(gateway.speech_output_mode, "proactive")
        self.assertEqual(
            gateway.speech_output_target_frames,
            gateway.PROACTIVE_OUTPUT_TARGET_FRAMES,
        )
        self.assertEqual(bytes(gateway.speech_output_pcm), pcm)
        self.assertEqual(bytes(gateway.speech_pcm), pcm)

    def test_speech_frame_counter_excludes_idle_frames(self):
        legacy_audio = (bytes(gateway.PCM_PACKET_BYTES),) * 2
        idle = (0, b"frame", 1, 1, legacy_audio, False)
        speech = (0, b"frame", 1, 1, legacy_audio, True)

        gateway.enqueue_av_frame(idle)
        gateway.enqueue_av_frame(speech)

        self.assertEqual(gateway.speech_frames_queued, 1)
        gateway.dequeue_av_frame()
        self.assertEqual(gateway.speech_frames_queued, 1)
        gateway.dequeue_av_frame()
        self.assertEqual(gateway.speech_frames_queued, 0)


class SemanticExpressionEnvelopeTests(unittest.TestCase):
    def setUp(self):
        gateway.expression_profile = "neutral"
        gateway.expression_gain = 0.0
        gateway.expression_target = 0.6
        gateway.expression_mouth_strength = 0.2
        gateway.expression_expires_at = 0.0
        gateway.expression_pending = None
        gateway.expression_after_speech = None
        gateway.expression_owner = "none"
        gateway.speech_output_active = False
        gateway.idle_expression_actions.clear()
        gateway.expression_timeline.clear()

    def test_expression_attacks_smoothly_over_five_frames(self):
        values = gateway._expression_frame_weights(100.0)
        self.assertEqual(len(values), gateway.CHUNK_SIZE)
        self.assertGreater(values[-1], values[0])
        self.assertLessEqual(values[-1], 0.6)

    def test_default_avatar_uses_visible_expression_source(self):
        gateway.expression_profile = "one_brow"
        gateway.expression_gain = 0.4
        gateway.expression_target = 0.7
        self.assertEqual(
            gateway._render_avatar_for_expression("xiaoya_locket"),
            "xiaoya_locket_expr_one_brow",
        )
        self.assertEqual(gateway._render_avatar_for_expression("xiaoya"), "xiaoya")

    def test_mild_everyday_expression_stays_on_base_portrait(self):
        gateway.expression_profile = "laugh"
        gateway.expression_gain = 0.2
        gateway.expression_target = 0.34
        self.assertEqual(
            gateway._render_avatar_for_expression("xiaoya_locket"),
            "xiaoya_locket",
        )

    def test_new_faces_stay_on_base_portrait(self):
        gateway.expression_profile = "side_eye"
        gateway.expression_gain = 0.8
        gateway.expression_target = 0.8
        self.assertEqual(
            gateway._render_avatar_for_expression("xiaoya_locket"),
            "xiaoya_locket",
        )
        gateway.expression_profile = "lip_bite"
        self.assertEqual(
            gateway._render_avatar_for_expression("xiaoya_locket"),
            "xiaoya_locket",
        )

    def test_profile_switch_uses_gradual_release_before_replacing_basis(self):
        gateway.expression_profile = "laugh"
        gateway.expression_gain = 0.5
        gateway.expression_target = 0.0
        gateway.expression_pending = ("one_brow", 0.45, 0.08, 900, "dialogue")
        gateway._expression_frame_weights(100.0)
        self.assertEqual(gateway.expression_profile, "laugh")
        self.assertGreater(gateway.expression_gain, 0.0)
        for _ in range(3):
            gateway._expression_frame_weights(100.0)
        self.assertEqual(gateway.expression_profile, "one_brow")
        self.assertGreater(gateway.expression_target, 0.0)

    def test_expression_timeline_follows_pcm_elapsed_time(self):
        gateway.expression_timeline.extend([
            (0, "shy", 0.62, 0.08, 1800, 1),
            (1200, "smirk", 0.74, 0.12, 1800, 2),
        ])
        gateway._apply_due_expressions(0)
        self.assertEqual(gateway.expression_profile, "shy")
        self.assertEqual(len(gateway.expression_timeline), 1)
        gateway.expression_gain = 0.0
        gateway._apply_due_expressions(1199)
        self.assertEqual(gateway.expression_profile, "shy")
        gateway._apply_due_expressions(1200)
        self.assertEqual(gateway.expression_profile, "smirk")
        self.assertFalse(gateway.expression_timeline)

    def test_idle_playful_faces_are_scheduled_often(self):
        import random as random_module

        random_module.seed(1)
        gateway.IDLE_EXPRESSION_MIN_SECONDS = 2.0
        gateway.IDLE_EXPRESSION_MAX_SECONDS = 4.5
        names: list[str] = []
        for index in range(80):
            gateway.idle_expression_last_name = names[-1] if names else ""
            gateway.idle_expression_actions.clear()
            gateway._schedule_idle_expression(1000.0 + index)
            names.append(gateway.idle_expression_last_name)
        playful = {
            "solo_wink", "playful_wink", "solo_pout", "thinking_pout",
            "tease_side_eye", "naughty_lip_bite", "sleepy_cute",
        }
        self.assertGreaterEqual(sum(name in playful for name in names), 55)
        self.assertIn("solo_wink", names)
        self.assertIn("solo_pout", names)
        self.assertIn("tease_side_eye", names)
        repeats = sum(
            left == right for left, right in zip(names, names[1:])
        )
        self.assertEqual(repeats, 0)

    def test_idle_expression_schedules_only_while_quiet(self):
        gateway.IDLE_EXPRESSION_ENABLED = True
        gateway.idle_expression_next_at = 99.0
        gateway.last_speech_input_at = 0.0
        gateway.last_user_voice_at = 0.0
        gateway.last_motion_audio_at = 0.0

        gateway._update_idle_expression(100.0, idle_allowed=True)

        self.assertGreater(gateway.idle_expression_sequences, 0)
        self.assertTrue(
            gateway.idle_expression_actions or gateway.expression_owner == "ambient"
        )

    def test_dialogue_expression_preempts_idle_choreography(self):
        gateway.expression_owner = "ambient"
        gateway.expression_profile = "pout"
        gateway.expression_gain = 0.4
        gateway.expression_target = 0.5
        gateway.idle_expression_actions.append(
            (100.0, "blink", "neutral", 0.0, 0.0, 0)
        )

        gateway._apply_expression("laugh", 0.6, 0.05, 1200)

        self.assertFalse(gateway.idle_expression_actions)
        self.assertEqual(gateway.expression_pending[-1], "dialogue")
        self.assertEqual(gateway.expression_target, 0.0)

    def test_lip_bite_waits_until_speech_ends(self):
        gateway.speech_output_active = True
        gateway.expression_gain = 0.0
        gateway._apply_expression("lip_bite", 0.78, 0.04, 2400)

        self.assertEqual(gateway.expression_profile, "smirk")
        self.assertEqual(gateway.expression_after_speech[0], "lip_bite")

        gateway.speech_output_active = False
        gateway._apply_deferred_silent_expression()
        self.assertEqual(gateway.expression_profile, "lip_bite")
        self.assertIsNone(gateway.expression_after_speech)

    def test_idle_lip_bite_is_not_remapped(self):
        gateway.speech_output_active = True
        gateway.expression_gain = 0.0
        gateway._apply_expression("lip_bite", 0.78, 0.04, 2400, owner="ambient")
        self.assertEqual(gateway.expression_profile, "lip_bite")
        self.assertIsNone(gateway.expression_after_speech)

    def test_expression_source_switch_uses_bounded_non_recursive_transition(self):
        gateway.expression_render_avatar = "xiaoya_locket"
        gateway.expression_previous_frame = bytes([10, 20, 30, 40])
        gateway.expression_transition_from_frame = None
        gateway.expression_transition_frames = 0
        gateway.expression_transition_total_frames = 0
        gateway.speech_output_active = True
        gateway.speech_playing = False
        gateway.speech_turn_active = False
        new_frame = bytes([200, 180, 160, 140])
        first = gateway._crossfade_expression_frame(
            new_frame, "xiaoya_locket_expr_smirk"
        )
        self.assertNotEqual(first, new_frame)
        self.assertGreater(first[0], 10)
        self.assertLess(first[0], 200)
        self.assertEqual(
            gateway.expression_transition_frames,
            gateway.EXPRESSION_TRANSITION_SPEECH_FRAMES - 1,
        )
        output = first
        for _ in range(gateway.EXPRESSION_TRANSITION_SPEECH_FRAMES - 1):
            output = gateway._crossfade_expression_frame(
                new_frame, "xiaoya_locket_expr_smirk"
            )
        self.assertEqual(output, new_frame)
        self.assertEqual(gateway.expression_transition_frames, 0)
        self.assertIsNone(gateway.expression_transition_from_frame)


if __name__ == "__main__":
    unittest.main()
