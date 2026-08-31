from __future__ import annotations

import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


PROXY = Path(__file__).resolve().parents[1] / "proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

from voxcpm_shared_client import (  # noqa: E402
    SharedVoxCPMClient,
    clone_pace_factor,
    decode_wav,
)


def _pcm16_wav(samples: np.ndarray, sample_rate: int = 48000) -> bytes:
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _pcm16_bytes(samples: np.ndarray) -> bytes:
    return np.clip(samples * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def _stream_response(status: int, headers=None, chunks=None, text: str = ""):
    response = MagicMock()
    response.status_code = status
    response.headers = headers or {}
    response.text = text
    response.iter_bytes.return_value = iter(chunks or [])
    context = MagicMock()
    context.__enter__.return_value = response
    context.__exit__.return_value = False
    return context


class ClonePaceTests(unittest.TestCase):
    def test_short_welcome_keeps_original_tempo(self) -> None:
        self.assertEqual(clone_pace_factor("诶，张三丰！刚进来就喊你一声哈", 3.56), 1.0)

    def test_rushed_news_is_slowed_toward_the_reference(self) -> None:
        text = (
            "刚出炉的考古大新闻，中埃联合考古队在埃及塞赫迈特神庙遗址挖到好东西了。"
            "除了泥砖建筑，还真发现了跟孟菲斯古城卜塔神庙有关的老石块和雕像碎片，"
            "连古埃及新王国时期的红酒酿造设施都跑出来露脸。"
            "这发现不光能补历史拼图，感觉两边合作真挺有味道。"
            "你们觉得这对以后解读金字塔周围的日常生活有啥实际帮助？"
        )
        factor = clone_pace_factor(text, 21.61)
        self.assertLess(factor, 0.95)
        self.assertGreaterEqual(factor, 0.86)

    def test_reference_paced_news_is_left_alone(self) -> None:
        text = "嗯……就在刚才，库克给苹果全体员工发了封告别信，说他今天最后一天当CEO了。"
        self.assertEqual(clone_pace_factor(text * 2, 17.52), 1.0)


class SharedVoxCPMClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.ref = Path(self.tmp.name) / "ref.wav"
        self.ref.write_bytes(_pcm16_wav(np.zeros(4800, dtype=np.float32)))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_decode_wav_roundtrip(self) -> None:
        payload = _pcm16_wav(np.array([0.0, 0.5, -0.25], dtype=np.float32), 16000)
        audio, sr = decode_wav(payload)
        self.assertEqual(sr, 16000)
        self.assertEqual(audio.shape[0], 3)
        self.assertAlmostEqual(float(audio[1]), 0.5, places=3)

    def test_empty_text_yields_nothing(self) -> None:
        client = SharedVoxCPMClient(
            base_url="http://127.0.0.1:10102",
            api_key="secret",
            ref_audio=str(self.ref),
        )
        self.assertEqual(list(client.stream_clone("  ")), [])

    def test_stream_route_buffers_the_sentence_before_playback_chunks(self) -> None:
        pcm = _pcm16_bytes(np.linspace(-0.2, 0.2, 9000, dtype=np.float32))
        seen_second_half = []

        def delayed_chunks():
            yield pcm[:2000]
            seen_second_half.append(False)
            yield pcm[2000:]
            seen_second_half.append(True)

        busy = _stream_response(429, {"Retry-After": "0"})
        ok = _stream_response(
            200,
            {"X-Sample-Rate": "48000", "X-VoxCPM-Format": "pcm_s16le"},
            delayed_chunks(),
        )
        client = SharedVoxCPMClient(
            base_url="http://127.0.0.1:10102",
            api_key="secret",
            ref_audio=str(self.ref),
            ref_text="参考转写",
            timeout_s=5,
        )
        with patch("voxcpm_shared_client.httpx.stream", side_effect=[busy, ok]) as stream:
            iterator = client.stream_clone("[laughing]你好。")
            first = next(iterator)
            self.assertEqual(seen_second_half, [False, True])
            chunks = [first, *iterator]
        self.assertEqual(stream.call_count, 2)
        self.assertGreaterEqual(len(chunks), 2)
        audio = np.concatenate([chunk[0] for chunk in chunks])
        self.assertEqual(chunks[0][1], 48000)
        self.assertEqual(audio.size, 9000)
        sent = stream.call_args.kwargs["data"]
        self.assertEqual(sent["text"], "[laughing]你好。")
        self.assertEqual(sent["prompt_text"], "参考转写")
        self.assertTrue(str(stream.call_args.args[1]).endswith("/v1/audio/speech/stream"))

    def test_missing_stream_route_falls_back_to_wav(self) -> None:
        speech = _pcm16_wav(np.linspace(-0.2, 0.2, 4800, dtype=np.float32), 48000)
        missing = _stream_response(404, text="not found")
        ok = MagicMock(status_code=200, headers={}, content=speech, text="")
        client = SharedVoxCPMClient(
            base_url="http://127.0.0.1:10102",
            api_key="secret",
            ref_audio=str(self.ref),
            ref_text="参考转写",
            timeout_s=5,
        )
        with patch("voxcpm_shared_client.httpx.stream", return_value=missing):
            with patch("voxcpm_shared_client.httpx.post", return_value=ok) as post:
                chunks = list(client.stream_clone("你好。"))
        self.assertFalse(client._use_stream)
        self.assertEqual(post.call_count, 1)
        self.assertGreaterEqual(len(chunks), 1)

    def test_fast_first_sentence_skips_prompt_text(self) -> None:
        pcm = _pcm16_bytes(np.linspace(-0.2, 0.2, 4800, dtype=np.float32))
        ok = _stream_response(200, {"X-Sample-Rate": "48000"}, [pcm])
        client = SharedVoxCPMClient(
            base_url="http://127.0.0.1:10102",
            api_key="secret",
            ref_audio=str(self.ref),
            ref_text="参考转写",
            timeout_s=5,
        )
        with patch("voxcpm_shared_client.httpx.stream", return_value=ok) as stream:
            list(client.stream_clone("[playful]嘿嘿，", fast=True))
        sent = stream.call_args.kwargs["data"]
        self.assertEqual(sent["text"], "[playful]嘿嘿，")
        self.assertNotIn("prompt_text", sent)

    def test_http_error_is_surfaced(self) -> None:
        client = SharedVoxCPMClient(
            base_url="http://127.0.0.1:10102",
            api_key="secret",
            ref_audio=str(self.ref),
            timeout_s=5,
        )
        err = _stream_response(500, text="boom")
        with patch("voxcpm_shared_client.httpx.stream", return_value=err):
            with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
                list(client.stream_clone("你好"))


if __name__ == "__main__":
    unittest.main()
