"""HTTP client for the local IndexTTS-2.5 worker."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator

import httpx
import numpy as np

from playback_policy import take_live_tts_sentence

LOG = logging.getLogger("speech_to_speech.TTS.qwen3_tts_handler")
DEFAULT_URL = "http://127.0.0.1:18782"
CHUNK_SAMPLES = 4096
NATIVE_SR = 22050


def play_reservoir_seconds(*, followup: bool = False) -> float:
    first = max(0.4, float(os.environ.get("INDEXTTS_PLAY_RESERVOIR_SECONDS", "0.8")))
    if not followup:
        return first
    later = float(os.environ.get("INDEXTTS_FOLLOWUP_RESERVOIR_SECONDS", "0.16"))
    return max(0.04, min(first, later))


def play_reservoir_samples(sample_rate: int, *, followup: bool = False) -> int:
    seconds = play_reservoir_seconds(followup=followup)
    return max(1, int(max(1, int(sample_rate or NATIVE_SR)) * seconds))


def live_tts_options(batch: bool) -> dict[str, bool]:
    """First live sentence keeps a start reservoir; later sentences do not."""

    interactive = not bool(batch)
    followup = interactive and take_live_tts_sentence() > 0
    return {"live": interactive, "followup": followup}


def pcm16_to_float(payload: bytes) -> np.ndarray:
    if not payload:
        return np.zeros(0, dtype=np.float32)
    usable = payload[: len(payload) - (len(payload) % 2)]
    return np.frombuffer(usable, dtype=np.int16).astype(np.float32) / 32768.0


def yield_audio_chunks(
    audio: np.ndarray, sample_rate: int
) -> Iterator[tuple[np.ndarray, int, None]]:
    pcm = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    for start in range(0, pcm.size, CHUNK_SAMPLES):
        yield pcm[start : start + CHUNK_SAMPLES], sample_rate, None


class IndexTTSClient:
    def __init__(
        self,
        base_url: str = "",
        ref_audio: str = "",
        ref_text: str = "",
        timeout_s: float = 180.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("INDEXTTS_URL") or DEFAULT_URL
        ).rstrip("/")
        self.timeout_s = float(os.environ.get("INDEXTTS_TIMEOUT_SECONDS") or timeout_s)
        self._ref_path = ""
        self._ref_text = ""
        self._ref_key = ""
        if ref_audio:
            self.set_reference(ref_audio, ref_text)

    def set_reference(self, ref_audio: str, ref_text: str = "") -> None:
        path = Path(ref_audio)
        key = f"{path}:{path.stat().st_mtime_ns if path.is_file() else 0}:{ref_text}"
        if key == self._ref_key and self._ref_path:
            return
        if not path.is_file():
            raise FileNotFoundError(f"IndexTTS reference audio missing: {path}")
        self._ref_path = str(path)
        self._ref_text = str(ref_text or "").strip()
        self._ref_key = key
        LOG.info("IndexTTS reference loaded path=%s", path)

    def wait_ready(self, tries: int = 180) -> None:
        last_error = "not started"
        for _ in range(max(1, tries)):
            try:
                response = httpx.get(f"{self.base_url}/healthz", timeout=2.0)
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("ok") and payload.get("ready"):
                        return
                    last_error = f"not ready: {payload}"
                else:
                    last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(2)
        raise TimeoutError(f"IndexTTS is not ready at {self.base_url}: {last_error}")

    def stream_clone(
        self,
        text: str,
        *,
        live: bool = True,
        followup: bool = False,
        emo_vector: list[float] | None = None,
        emo_alpha: float = 0.2,
        use_emo_text: bool = False,
        emo_text: str = "",
        duration_factor: float = 1.0,
        emo_audio: str | None = None,
        lang: str = "ZH",
    ) -> Iterator[tuple[np.ndarray, int, None]]:
        spoken = str(text or "").strip()
        if not spoken:
            return
        if not self._ref_path:
            raise RuntimeError("IndexTTS reference audio is not loaded")
        fields: dict[str, str] = {
            "text": spoken,
            "lang": lang,
            "ref_path": self._ref_path,
            "emo_vector": json.dumps(list(emo_vector or [])),
            "emo_alpha": str(emo_alpha),
            "use_emo_text": "1" if use_emo_text else "0",
            "emo_text": str(emo_text or ""),
            "duration_factor": str(duration_factor),
            "interval_silence": "0",
        }
        files: dict[str, Any] | None = None
        file_handle = None
        if emo_audio and Path(emo_audio).is_file():
            file_handle = Path(emo_audio).open("rb")
            files = {"emo_audio": (Path(emo_audio).name, file_handle, "audio/wav")}
        deadline = time.monotonic() + self.timeout_s
        last_error = "no attempt"
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"IndexTTS stream queue timed out: {last_error}")
                yielded = False
                try:
                    with httpx.stream(
                        "POST",
                        f"{self.base_url}/v1/audio/speech/stream",
                        data=fields,
                        files=files,
                        timeout=httpx.Timeout(max(5.0, remaining), connect=5.0),
                    ) as response:
                        if response.status_code == 429:
                            retry_after = response.headers.get("Retry-After", "2")
                            try:
                                wait_s = max(0.2, float(retry_after))
                            except ValueError:
                                wait_s = 2.0
                            last_error = "worker busy"
                            LOG.info("IndexTTS busy; waiting %.1fs", wait_s)
                            time.sleep(min(wait_s, max(0.2, remaining)))
                            continue
                        if response.status_code >= 400:
                            raise RuntimeError(
                                f"IndexTTS HTTP {response.status_code}: {response.text[:400]}"
                            )
                        sample_rate = int(response.headers.get("X-Sample-Rate") or NATIVE_SR)
                        leftover = b""
                        held: list[np.ndarray] = []
                        held_samples = 0
                        started_live = False
                        reservoir = (
                            play_reservoir_samples(sample_rate, followup=followup)
                            if live
                            else 10**12
                        )
                        for raw in response.iter_bytes():
                            if not raw:
                                continue
                            leftover += raw
                            usable = leftover[: len(leftover) - (len(leftover) % 2)]
                            leftover = leftover[len(usable) :]
                            audio = pcm16_to_float(usable)
                            if not audio.size:
                                continue
                            if started_live:
                                yielded = True
                                yield from yield_audio_chunks(audio, sample_rate)
                                continue
                            held.append(audio)
                            held_samples += audio.size
                            if held_samples < reservoir:
                                continue
                            full = held[0] if len(held) == 1 else np.concatenate(held)
                            held.clear()
                            started_live = True
                            LOG.info(
                                "IndexTTS live yield after %.2fs reservoir followup=%s",
                                held_samples / float(sample_rate or 1),
                                followup,
                            )
                            yielded = True
                            yield from yield_audio_chunks(full, sample_rate)
                        if held:
                            full = held[0] if len(held) == 1 else np.concatenate(held)
                            yielded = True
                            yield from yield_audio_chunks(full, sample_rate)
                        elif not started_live:
                            raise RuntimeError("IndexTTS stream returned empty audio")
                except httpx.HTTPError as exc:
                    last_error = str(exc)
                    time.sleep(min(2.0, max(0.2, remaining)))
                    continue
                if not yielded:
                    raise RuntimeError("IndexTTS stream returned empty audio")
                return
        finally:
            if file_handle is not None:
                file_handle.close()
