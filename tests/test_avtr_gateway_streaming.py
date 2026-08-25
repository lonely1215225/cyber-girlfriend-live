import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proxy"))

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


if __name__ == "__main__":
    unittest.main()
