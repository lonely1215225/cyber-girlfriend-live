#!/usr/bin/env python3
"""Run speech-to-speech with a server-local PCM tee for AVTR-1.

The realtime API normally sends generated PCM to the browser.  Sending that
PCM back over public HTTP to animate AVTR-1 makes lip sync depend on a second
network round trip. This wrapper mirrors the same audio events to the local
avatar gateway with a short preroll. The gateway owns realtime playback pacing;
the tee sends generated audio ahead as soon as it exists so GPU jitter cannot
starve the renderer between 100 ms packets.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
import time

import httpx
import uvicorn

from speech_to_speech.api.openai_realtime import websocket_router

from tiered_memory import cancel_semantic_refinements, install_tiered_memory

LOG = logging.getLogger("speech_to_speech.avatar_tee")
LOG.setLevel(logging.INFO)
logging.getLogger("speech_to_speech.TTS.qwen3_tts_handler").setLevel(logging.INFO)
# The local pacer posts ten small chunks per second; per-request INFO logs would
# drown out the useful latency and pipeline messages.
logging.getLogger("httpx").setLevel(logging.WARNING)
_uvicorn_config_init = uvicorn.Config.__init__
install_tiered_memory()

if os.environ.get("STT_BACKEND", "sensevoice") == "sensevoice":
    from sensevoice_stt import install_sensevoice_adapter

    install_sensevoice_adapter()

if os.environ.get("TTS_EMOTION_ENABLED", "1") != "0":
    from emotion_aware_tts import install_emotion_aware_tts

    install_emotion_aware_tts()


def _quiet_uvicorn_config(self, *args, **kwargs):
    kwargs.setdefault("access_log", False)
    _uvicorn_config_init(self, *args, **kwargs)


uvicorn.Config.__init__ = _quiet_uvicorn_config
TEE_URL = os.environ.get("AVTR1_LOCAL_TEE_URL", "").rstrip("/")
SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
PREROLL_MS = max(420, int(os.environ.get("AVATAR_TEE_PREROLL_MS", "800")))
PREROLL_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * PREROLL_MS / 1000)
PACE_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.1)


class LocalAvatarTee:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.pending = bytearray()
        self.generation = 0
        self.done = False
        self.complete_flush = False
        self.pump_task: asyncio.Task | None = None
        self.client: httpx.AsyncClient | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.last_feed_at: float | None = None
        self.audio_bytes = 0
        self.chunk_count = 0
        self.max_feed_gap_ms = 0.0

    def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=httpx.Timeout(2.0))
        return self.client

    def feed(self, pcm: bytes) -> None:
        if not pcm:
            return
        now = time.monotonic()
        if self.started_at is None:
            self.started_at = now
            self.finished_at = None
            self.audio_bytes = 0
            self.chunk_count = 0
            self.max_feed_gap_ms = 0.0
        if self.last_feed_at is not None:
            self.max_feed_gap_ms = max(
                self.max_feed_gap_ms, (now - self.last_feed_at) * 1000.0
            )
        self.last_feed_at = now
        self.audio_bytes += len(pcm)
        self.chunk_count += 1
        self.pending.extend(pcm)
        if (
            not self.complete_flush
            and self.pump_task is None
            and len(self.pending) >= PREROLL_BYTES
        ):
            self._start_pump()

    def finish(self, *, complete: bool = False) -> None:
        self.done = True
        self.complete_flush = bool(complete)
        if self.started_at is not None and self.finished_at is None:
            self.finished_at = time.monotonic()
        if self.pending and self.pump_task is None:
            self._start_pump()

    async def interrupt(self) -> None:
        self.generation += 1
        self.pending.clear()
        self.done = False
        self._reset_metrics()
        task = self.pump_task
        self.pump_task = None
        if task is not None:
            task.cancel()
        try:
            await self._client().post(f"{self.base_url}/interrupt")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("AVTR-1 local interrupt failed: %s", exc)

    def _start_pump(self) -> None:
        generation = self.generation
        self.pump_task = asyncio.create_task(self._pump(generation))

    def _metrics(self) -> dict[str, float | int]:
        audio_ms = self.audio_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE) * 1000.0
        end = self.finished_at or time.monotonic()
        generation_ms = max(0.0, (end - self.started_at) * 1000.0) if self.started_at else 0.0
        return {
            "audio_ms": round(audio_ms, 3),
            "generation_ms": round(generation_ms, 3),
            "rtf": round(audio_ms / generation_ms, 3) if generation_ms > 0 else 0.0,
            "chunks": self.chunk_count,
            "max_gap_ms": round(self.max_feed_gap_ms, 3),
        }

    def _reset_metrics(self) -> None:
        self.started_at = None
        self.finished_at = None
        self.last_feed_at = None
        self.audio_bytes = 0
        self.chunk_count = 0
        self.max_feed_gap_ms = 0.0
        self.complete_flush = False

    async def _pump(self, generation: int) -> None:
        first = True
        try:
            while generation == self.generation:
                threshold = PREROLL_BYTES if first else PACE_BYTES
                if len(self.pending) >= threshold or (self.done and self.pending):
                    if first:
                        # Send a bounded first packet. A large streaming TTS
                        # delta must not overflow the renderer's live buffer and
                        # silently discard the beginning of the sentence.
                        size = (
                            len(self.pending)
                            if self.complete_flush and self.done
                            else min(PREROLL_BYTES, len(self.pending))
                        )
                        first = False
                    else:
                        size = min(PACE_BYTES, len(self.pending))
                    payload = bytes(self.pending[:size])
                    del self.pending[:size]
                    try:
                        await self._client().post(
                            f"{self.base_url}/audio-chunk",
                            content=payload,
                            headers={"Content-Type": "application/octet-stream"},
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("AVTR-1 local audio tee failed: %s", exc)

                    # Do not sleep for the audio duration here. AVTR's FLV pacer
                    # already plays at realtime speed; sending ahead builds a
                    # bounded jitter buffer instead of causing underruns.
                    await asyncio.sleep(0)
                    continue

                if self.done and not self.pending:
                    self.done = False
                    metrics = self._metrics()
                    try:
                        await self._client().post(
                            f"{self.base_url}/audio-finish",
                            params={key: str(value) for key, value in metrics.items()},
                        )
                        LOG.info(
                            "TTS turn audio=%.0fms generation=%.0fms realtime=%.2fx "
                            "chunks=%d max_gap=%.0fms",
                            metrics["audio_ms"],
                            metrics["generation_ms"],
                            metrics["rtf"],
                            metrics["chunks"],
                            metrics["max_gap_ms"],
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("AVTR-1 local audio finish failed: %s", exc)
                    self._reset_metrics()
                    return
                await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            pass
        finally:
            if generation == self.generation and self.pump_task is asyncio.current_task():
                self.pump_task = None


if TEE_URL:
    tee = LocalAvatarTee(TEE_URL)
    original_send_events = websocket_router._send_events

    async def send_events_with_avatar_tee(ws, events):
        query_params = getattr(ws, "query_params", {})
        is_preview = str(query_params.get("preview", "")) == "1" if hasattr(query_params, "get") else False
        # Proactive news has no conversational latency requirement. Buffer its
        # entire synthesized turn before AVTR playback so multi-sentence TTS
        # stalls can never drain the live reservoir halfway through a report.
        complete_audio = (
            str(query_params.get("complete_audio", "")) == "1"
            if hasattr(query_params, "get") else False
        )
        if is_preview:
            await original_send_events(ws, events)
            return
        for event in events:
            event_type = getattr(event, "type", "")
            if event_type == "response.created" and complete_audio:
                # Clear a cancelled/incomplete prior buffered turn before
                # accepting the new one. The room permits only one bot speaker.
                await tee.interrupt()
                tee.complete_flush = True
            elif event_type in ("response.audio.delta", "response.output_audio.delta"):
                delta = getattr(event, "delta", "")
                if delta:
                    try:
                        tee.feed(base64.b64decode(delta))
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("Invalid realtime PCM delta: %s", exc)
            elif event_type in ("response.audio.done", "response.output_audio.done"):
                # Realtime mode finishes each sentence promptly. Completeness
                # mode waits for response.done so every sentence is present.
                if not complete_audio:
                    tee.finish()
            elif event_type == "input_audio_buffer.speech_started":
                # Semantic memory uses the conversation model only while idle.
                # Cancel it at VAD onset, before STT can submit the next LLM turn.
                cancel_semantic_refinements()
                await tee.interrupt()
            elif event_type == "response.done":
                response = getattr(event, "response", None)
                if getattr(response, "status", "") == "cancelled":
                    await tee.interrupt()
                else:
                    tee.finish(complete=complete_audio)
        await original_send_events(ws, events)

    websocket_router._send_events = send_events_with_avatar_tee
    LOG.info("AVTR-1 local audio tee enabled: %s", TEE_URL)


from speech_to_speech.s2s_pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
