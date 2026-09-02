from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from tests.import_paths import add_service_paths

add_service_paths()

from indextts_client import (  # noqa: E402
    IndexTTSClient,
    live_infer_options,
    live_tts_options,
    pcm16_to_float,
    play_reservoir_seconds,
)
from playback_policy import begin_live_tts_turn, set_batch_tts  # noqa: E402


def _pcm16_bytes(audio: np.ndarray) -> bytes:
    return np.clip(audio * 32767.0, -32767, 32767).astype(np.int16).tobytes()


class _FakeResponse:
    def __init__(self, status_code: int, body: bytes = b"", headers=None, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {"X-Sample-Rate": "22050"}
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_bytes(self):
        yield self._body


class IndexTTSClientTests(unittest.TestCase):
    def setUp(self) -> None:
        set_batch_tts(False)
        begin_live_tts_turn()

    def test_live_infer_uses_one_beam_and_short_segments(self) -> None:
        with patch.dict("os.environ", {
            "INDEXTTS_PLAY_RESERVOIR_SECONDS": "0.32",
            "INDEXTTS_NUM_BEAMS": "1",
            "INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT": "28",
        }):
            options = live_infer_options()
            self.assertEqual(options["num_beams"], 1)
            self.assertEqual(options["max_text_tokens_per_segment"], 28)
            self.assertEqual(play_reservoir_seconds(followup=False), 0.32)

    def test_live_options_mark_followup_after_first_sentence(self) -> None:
        begin_live_tts_turn()
        self.assertEqual(live_tts_options(False), {"live": True, "followup": False})
        self.assertEqual(live_tts_options(False), {"live": True, "followup": True})
        begin_live_tts_turn()
        self.assertEqual(live_tts_options(True), {"live": False, "followup": False})

    def test_empty_text_yields_nothing(self) -> None:
        client = IndexTTSClient(base_url="http://127.0.0.1:18782")
        client._ref_path = "/tmp/ref.wav"
        self.assertEqual(list(client.stream_clone("")), [])

    def test_live_stream_holds_first_reservoir(self) -> None:
        pcm = _pcm16_bytes(np.linspace(-0.2, 0.2, 22050, dtype=np.float32))
        client = IndexTTSClient(base_url="http://127.0.0.1:18782")
        client._ref_path = "/tmp/ref.wav"
        ok = _FakeResponse(200, pcm)
        with patch("indextts_client.play_reservoir_samples", return_value=8000):
            with patch("indextts_client.httpx.stream", return_value=ok):
                chunks = list(client.stream_clone("你好呀。", live=True, followup=False))
        self.assertTrue(chunks)
        audio = np.concatenate([item[0] for item in chunks])
        self.assertEqual(chunks[0][1], 22050)
        self.assertGreaterEqual(audio.size, 8000)

    def test_busy_worker_retries(self) -> None:
        pcm = _pcm16_bytes(np.linspace(-0.2, 0.2, 4096, dtype=np.float32))
        client = IndexTTSClient(base_url="http://127.0.0.1:18782", timeout_s=5)
        client._ref_path = "/tmp/ref.wav"
        busy = _FakeResponse(429, b"", {"Retry-After": "0.2"})
        ok = _FakeResponse(200, pcm)
        with patch("indextts_client.httpx.stream", side_effect=[busy, ok]) as stream:
            chunks = list(client.stream_clone("嗯，我在。", live=True, followup=True))
        self.assertEqual(stream.call_count, 2)
        self.assertTrue(chunks)

    def test_loading_worker_retries(self) -> None:
        pcm = _pcm16_bytes(np.linspace(-0.2, 0.2, 4096, dtype=np.float32))
        client = IndexTTSClient(base_url="http://127.0.0.1:18782", timeout_s=5)
        client._ref_path = "/tmp/ref.wav"
        loading = _FakeResponse(503, b"", {"Retry-After": "0.2"})
        ok = _FakeResponse(200, pcm)
        with patch("indextts_client.httpx.stream", side_effect=[loading, ok]) as stream:
            chunks = list(client.stream_clone("嗯，我在。", live=True, followup=True))
        self.assertEqual(stream.call_count, 2)
        self.assertTrue(chunks)

    def test_pcm16_roundtrip(self) -> None:
        source = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        recovered = pcm16_to_float(_pcm16_bytes(source))
        self.assertTrue(np.allclose(recovered, source, atol=2e-4))


if __name__ == "__main__":
    unittest.main()
