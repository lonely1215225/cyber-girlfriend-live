"""HTTP client for the local Fish Speech S2 Pro API server."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import httpx
import numpy as np
import ormsgpack

LOG = logging.getLogger("speech_to_speech.TTS.qwen3_tts_handler")
FISH_SAMPLE_RATE = 44100


class FishS2Client:
    def __init__(
        self,
        base_url: str,
        ref_audio: str = "",
        ref_text: str = "",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._audio_bytes = b""
        self._ref_text = ""
        self._ref_key = ""
        if ref_audio:
            self.set_reference(ref_audio, ref_text)

    def set_reference(self, ref_audio: str, ref_text: str) -> None:
        path = Path(ref_audio)
        key = f"{path}:{path.stat().st_mtime_ns if path.is_file() else 0}:{ref_text}"
        if key == self._ref_key and self._audio_bytes:
            return
        if not path.is_file():
            raise FileNotFoundError(f"Fish reference audio missing: {path}")
        self._audio_bytes = path.read_bytes()
        self._ref_text = str(ref_text or "").strip()
        self._ref_key = key
        LOG.info("Fish S2 reference loaded path=%s bytes=%d", path, len(self._audio_bytes))

    def wait_ready(self, tries: int = 90) -> None:
        last_error = "not started"
        for _ in range(max(1, tries)):
            try:
                response = httpx.get(f"{self.base_url}/v1/health", timeout=2.0)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(2)
        raise TimeoutError(f"Fish S2 is not ready at {self.base_url}: {last_error}")

    def stream_clone(self, text: str) -> Iterator[tuple[np.ndarray, int, None]]:
        spoken = str(text or "").strip()
        if not spoken:
            return
        if not self._audio_bytes:
            raise RuntimeError("Fish S2 reference audio is not loaded")
        payload = {
            "text": spoken,
            "references": [{"audio": self._audio_bytes, "text": self._ref_text}],
            "format": "wav",
            "streaming": True,
            "use_memory_cache": "on",
            "chunk_length": 200,
            "max_new_tokens": 1024,
            "top_p": 0.8,
            "temperature": 0.8,
            "repetition_penalty": 1.1,
            "normalize": True,
        }
        with httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=10.0)) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/v1/tts",
                content=ormsgpack.packb(payload),
                headers={
                    "content-type": "application/msgpack",
                    "accept": "application/octet-stream",
                },
            ) as response:
                if response.status_code >= 400:
                    body = response.read()
                    raise RuntimeError(
                        f"Fish S2 HTTP {response.status_code}: {body[:500]!r}"
                    )
                pending = bytearray()
                header_skipped = False
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    pending.extend(chunk)
                    if not header_skipped:
                        if len(pending) < 44 or pending[:4] != b"RIFF":
                            if len(pending) >= 4 and pending[:4] != b"RIFF":
                                header_skipped = True
                            else:
                                continue
                        else:
                            del pending[:44]
                            header_skipped = True
                    usable = len(pending) - (len(pending) % 2)
                    if usable < 4096:
                        continue
                    yield self._pcm16_chunk(pending, usable)
                usable = len(pending) - (len(pending) % 2)
                if usable:
                    yield self._pcm16_chunk(pending, usable)

    @staticmethod
    def _pcm16_chunk(pending: bytearray, usable: int) -> tuple[np.ndarray, int, None]:
        pcm = np.frombuffer(bytes(pending[:usable]), dtype=np.int16)
        del pending[:usable]
        return pcm.astype(np.float32) / 32768.0, FISH_SAMPLE_RATE, None
