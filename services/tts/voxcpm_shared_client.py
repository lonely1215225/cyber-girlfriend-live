"""HTTP client for the already-running localization VoxCPM2 worker.

This process never loads VoxCPM weights. It queues on the existing
inference lock (HTTP 429). Dialogue uses ``/v1/audio/speech/stream`` so
the worker can emit PCM while cloning. VoxCPM2 clones at about 0.55x
realtime: generation is faster than playback, not slower. Playing the
first 0.4s chunk used to starve the avatar reservoir, so the mouth and
picture hitched. Dialogue now holds a short reservoir and then forwards
the rest of the stream; news still waits for the whole clip so it can
match the reference pace. The original WAV route remains the fallback.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
import wave
from io import BytesIO
from pathlib import Path
from typing import Iterator

import httpx
import numpy as np

LOG = logging.getLogger("speech_to_speech.TTS.qwen3_tts_handler")
DEFAULT_URL = "http://127.0.0.1:10102"
CHUNK_SAMPLES = 4096
WORKER_CMDLINE = b"voxcpm_server.py"
_HANZI_RE = re.compile(r"[\u3400-\u9fff]")
# The live clone prompt speaks about 4.2 hanzi/s. Long VoxCPM batches often
# land at 5.3-6.3, which sounds rushed next to that reference.
TARGET_HANZI_PER_SEC = max(
    3.2, float(os.environ.get("VOXCPM_TARGET_HANZI_PER_SEC", "4.5"))
)
PACE_FAST_THRESHOLD = max(
    TARGET_HANZI_PER_SEC,
    float(os.environ.get("VOXCPM_PACE_FAST_THRESHOLD", "5.0")),
)
MIN_ATEMPO = min(0.98, max(0.80, float(os.environ.get("VOXCPM_MIN_ATEMPO", "0.86"))))
MIN_PACE_HANZI = max(16, int(os.environ.get("VOXCPM_PACE_MIN_HANZI", "24")))
DIALOGUE_TIMESTEPS = max(
    4, min(20, int(os.environ.get("VOXCPM_DIALOGUE_TIMESTEPS", "8")))
)


def play_reservoir_samples(sample_rate: int) -> int:
    """How much PCM to hold before live dialogue playback starts."""

    seconds = max(0.4, float(os.environ.get("VOXCPM_PLAY_RESERVOIR_SECONDS", "1.2")))
    return max(1, int(max(1, int(sample_rate or 48000)) * seconds))


class StreamUnavailable(RuntimeError):
    """The shared worker does not expose the additive streaming route."""


def resolve_voxcpm_api_key() -> str:
    for name in ("VOXCPM_API_KEY", "VOXCPM_SHARED_API_KEY", "LOCALIZATION_GPU_API_KEY"):
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    proc = Path("/proc")
    if not proc.is_dir():
        return ""
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if WORKER_CMDLINE not in cmdline:
            continue
        try:
            raw = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        found: dict[str, str] = {}
        for item in raw:
            if b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            try:
                found[key.decode()] = value.decode()
            except UnicodeDecodeError:
                continue
        secret = (
            found.get("VOXCPM_API_KEY")
            or found.get("LOCALIZATION_GPU_API_KEY")
            or ""
        ).strip()
        if secret:
            return secret
    return ""


def decode_wav(payload: bytes) -> tuple[np.ndarray, int]:
    with wave.open(BytesIO(payload), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width == 2:
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        pcm = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"unsupported VoxCPM WAV sample width {width}")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, sample_rate


def spoken_hanzi_count(text: str) -> int:
    return len(_HANZI_RE.findall(re.sub(r"\[[^\]]*\]", "", str(text or ""))))


def clone_pace_factor(text: str, duration_s: float) -> float:
    """Slow a rushed long clone toward the reference speaking rate.

    Short welcomes already match the prompt and must stay untouched. Only
    clips that are both long and clearly faster than the reference are
    stretched, and never below MIN_ATEMPO, so pitch-preserving atempo
    cannot turn speech syrupy.
    """
    hanzi = spoken_hanzi_count(text)
    if hanzi < MIN_PACE_HANZI or duration_s <= 0.2:
        return 1.0
    rate = hanzi / duration_s
    if rate <= PACE_FAST_THRESHOLD:
        return 1.0
    return max(MIN_ATEMPO, TARGET_HANZI_PER_SEC / rate)


def time_stretch(audio: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    """Pitch-preserving tempo change. factor < 1 slows speech."""
    if audio.size == 0 or abs(factor - 1.0) < 0.02:
        return audio
    pcm = np.ascontiguousarray(audio, dtype=np.float32)
    try:
        done = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "f32le",
                "-ar",
                str(int(sample_rate)),
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-filter:a",
                f"atempo={factor:.4f}",
                "-f",
                "f32le",
                "pipe:1",
            ],
            input=pcm.tobytes(),
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return audio
    if done.returncode != 0 or not done.stdout:
        return audio
    stretched = np.frombuffer(done.stdout, dtype=np.float32)
    return stretched if stretched.size else audio


def match_reference_pace(
    audio: np.ndarray, sample_rate: int, text: str
) -> np.ndarray:
    duration_s = float(audio.size) / float(sample_rate or 1)
    factor = clone_pace_factor(text, duration_s)
    if factor >= 0.99:
        return audio
    stretched = time_stretch(audio, sample_rate, factor)
    if stretched.size <= audio.size:
        return audio
    LOG.info(
        "VoxCPM pace hanzi=%s dur=%.2fs factor=%.2f -> %.2fs",
        spoken_hanzi_count(text),
        duration_s,
        factor,
        stretched.size / float(sample_rate or 1),
    )
    return stretched


def yield_audio_chunks(
    audio: np.ndarray, sample_rate: int
) -> Iterator[tuple[np.ndarray, int, None]]:
    for start in range(0, audio.size, CHUNK_SAMPLES):
        yield audio[start : start + CHUNK_SAMPLES], sample_rate, None


def pcm16_to_float(payload: bytes) -> np.ndarray:
    if not payload:
        return np.zeros(0, dtype=np.float32)
    usable = payload[: len(payload) - (len(payload) % 2)]
    return np.frombuffer(usable, dtype=np.int16).astype(np.float32) / 32768.0


class SharedVoxCPMClient:
    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        ref_audio: str = "",
        ref_text: str = "",
        timeout_s: float = 180.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("VOXCPM_SHARED_URL") or DEFAULT_URL).rstrip("/")
        self.api_key = api_key or resolve_voxcpm_api_key()
        self.timeout_s = float(os.environ.get("VOXCPM_TIMEOUT_SECONDS") or timeout_s)
        self._ref_path = ""
        self._ref_text = ""
        self._ref_key = ""
        self._use_stream = True
        if ref_audio:
            self.set_reference(ref_audio, ref_text)

    @property
    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": "Bearer " + self.api_key}

    def set_reference(self, ref_audio: str, ref_text: str) -> None:
        path = Path(ref_audio)
        key = f"{path}:{path.stat().st_mtime_ns if path.is_file() else 0}:{ref_text}"
        if key == self._ref_key and self._ref_path:
            return
        if not path.is_file():
            raise FileNotFoundError(f"VoxCPM reference audio missing: {path}")
        self._ref_path = str(path)
        self._ref_text = str(ref_text or "").strip()
        self._ref_key = key
        LOG.info("Shared VoxCPM reference path=%s", path)

    def wait_ready(self, tries: int = 30) -> None:
        last_error = "not started"
        for _ in range(max(1, tries)):
            try:
                response = httpx.get(
                    f"{self.base_url}/healthz",
                    headers=self.headers,
                    timeout=2.0,
                )
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("ok") and payload.get("ready"):
                        if payload.get("streaming"):
                            LOG.info("Shared VoxCPM streaming route is advertised")
                        return
                    last_error = f"not ready: {payload}"
                else:
                    last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(2)
        raise TimeoutError(f"shared VoxCPM is not ready at {self.base_url}: {last_error}")

    def stream_clone(
        self, text: str, *, fast: bool = False, live: bool = True
    ) -> Iterator[tuple[np.ndarray, int, None]]:
        spoken = str(text or "").strip()
        if not spoken:
            return
        if not self._ref_path:
            raise RuntimeError("VoxCPM reference audio is not loaded")
        if not self.api_key:
            raise RuntimeError("VoxCPM API key is missing")
        if self._use_stream:
            try:
                LOG.info("VoxCPM clone fast=%s live=%s", fast, live)
                yield from self._queued_stream(spoken, fast=fast, live=live)
                return
            except StreamUnavailable:
                self._use_stream = False
                LOG.info("Shared VoxCPM has no stream route; falling back to WAV")
        wav = self._queued_speech(spoken, fast=fast)
        audio, sample_rate = decode_wav(wav)
        if audio.size == 0:
            raise RuntimeError("VoxCPM returned empty audio")
        yield from yield_audio_chunks(
            match_reference_pace(audio, sample_rate, spoken), sample_rate
        )

    def _form_fields(self, text: str, *, fast: bool) -> dict[str, str]:
        fields = {"text": text}
        if self._ref_text and not fast:
            fields["prompt_text"] = self._ref_text
        if fast:
            fields["inference_timesteps"] = str(DIALOGUE_TIMESTEPS)
        return fields

    def _queued_stream(
        self, text: str, *, fast: bool = False, live: bool = True
    ) -> Iterator[tuple[np.ndarray, int, None]]:
        deadline = time.monotonic() + self.timeout_s
        last_error = "no attempt"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"VoxCPM stream queue timed out: {last_error}")
            yielded = False
            try:
                with Path(self._ref_path).open("rb") as handle:
                    with httpx.stream(
                        "POST",
                        f"{self.base_url}/v1/audio/speech/stream",
                        headers=self.headers,
                        data=self._form_fields(text, fast=fast),
                        files={"reference": (Path(self._ref_path).name, handle, "audio/wav")},
                        timeout=httpx.Timeout(max(5.0, remaining), connect=5.0),
                    ) as response:
                        if response.status_code == 404:
                            raise StreamUnavailable("stream route missing")
                        if response.status_code == 429:
                            retry_after = response.headers.get("Retry-After", "2")
                            try:
                                wait_s = max(0.2, float(retry_after))
                            except ValueError:
                                wait_s = 2.0
                            last_error = "worker busy"
                            LOG.info("Shared VoxCPM busy; waiting %.1fs", wait_s)
                            time.sleep(min(wait_s, max(0.2, remaining)))
                            continue
                        if response.status_code >= 400:
                            raise RuntimeError(
                                f"VoxCPM HTTP {response.status_code}: {response.text[:400]}"
                            )
                        sample_rate = int(response.headers.get("X-Sample-Rate") or 48000)
                        leftover = b""
                        held: list[np.ndarray] = []
                        held_samples = 0
                        started_live = False
                        reservoir = play_reservoir_samples(sample_rate) if live else 10**12
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
                                "VoxCPM live yield after %.2fs reservoir",
                                held_samples / float(sample_rate or 1),
                            )
                            yielded = True
                            yield from yield_audio_chunks(full, sample_rate)
                        if held:
                            full = held[0] if len(held) == 1 else np.concatenate(held)
                            if not started_live:
                                full = match_reference_pace(full, sample_rate, text)
                            yielded = True
                            yield from yield_audio_chunks(full, sample_rate)
                        elif not started_live:
                            raise RuntimeError("VoxCPM stream returned empty audio")
            except StreamUnavailable:
                raise
            except httpx.HTTPError as exc:
                last_error = str(exc)
                time.sleep(min(2.0, max(0.2, remaining)))
                continue
            if not yielded:
                raise RuntimeError("VoxCPM stream returned empty audio")
            return

    def _queued_speech(self, text: str, *, fast: bool = False) -> bytes:
        deadline = time.monotonic() + self.timeout_s
        last_error = "no attempt"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"VoxCPM queue timed out: {last_error}")
            try:
                with Path(self._ref_path).open("rb") as handle:
                    response = httpx.post(
                        f"{self.base_url}/v1/audio/speech",
                        headers=self.headers,
                        data=self._form_fields(text, fast=fast),
                        files={"reference": (Path(self._ref_path).name, handle, "audio/wav")},
                        timeout=httpx.Timeout(max(5.0, remaining), connect=5.0),
                    )
            except httpx.HTTPError as exc:
                last_error = str(exc)
                time.sleep(min(2.0, max(0.2, remaining)))
                continue
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "2")
                try:
                    wait_s = max(0.2, float(retry_after))
                except ValueError:
                    wait_s = 2.0
                last_error = "worker busy"
                LOG.info("Shared VoxCPM busy; waiting %.1fs", wait_s)
                time.sleep(min(wait_s, max(0.2, remaining)))
                continue
            if response.status_code >= 400:
                raise RuntimeError(
                    f"VoxCPM HTTP {response.status_code}: {response.text[:400]}"
                )
            if len(response.content) < 256:
                raise RuntimeError("VoxCPM returned an empty audio clip")
            return response.content
