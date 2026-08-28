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
from expression_director import begin_delivery_response, clear_delivery_state, cues_after

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
# Uploading to the local gateway and starting public playback are deliberately
# separate watermarks. The gateway owns the safe playback reservoir; holding
# the first PCM here as well made the safety buffer run twice before AVTR could
# even start rendering.
PREROLL_MS = max(240, int(os.environ.get("AVATAR_TEE_UPLOAD_PREROLL_MS", "320")))
PREROLL_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * PREROLL_MS / 1000)
PACE_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.1)
SEGMENT_GAP_SECONDS = max(
    0.9, float(os.environ.get("AVATAR_TEE_SEGMENT_GAP_MS", "1200")) / 1000.0
)


class LocalAvatarTee:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.pending = bytearray()
        self.generation = 0
        self.done = False
        self.complete_flush = False
        self.playback_mode = "interactive"
        self.pump_task: asyncio.Task | None = None
        self.client: httpx.AsyncClient | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.last_feed_at: float | None = None
        self.audio_bytes = 0
        self.chunk_count = 0
        self.max_feed_gap_ms = 0.0
        self.expression_sequence = 0
        self.expression_claimed = False
        self.consecutive_post_failures = 0
        self.active_connection_id: int | None = None
        self.segment_gap_task: asyncio.Task | None = None

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
        if self.segment_gap_task is not None:
            self.segment_gap_task.cancel()
        generation = self.generation
        marker = self.last_feed_at
        self.segment_gap_task = asyncio.create_task(
            self._finish_segment_after_gap(generation, marker)
        )
        if (
            not self.complete_flush
            and self.pump_task is None
            and len(self.pending) >= PREROLL_BYTES
        ):
            self._start_pump()

    async def _finish_segment_after_gap(
        self, generation: int, marker: float | None
    ) -> None:
        """Close a spoken segment while a slow cloud continuation is pending."""
        try:
            await asyncio.sleep(SEGMENT_GAP_SECONDS)
            if (
                generation == self.generation
                and marker is not None
                and self.last_feed_at == marker
                and not self.done
            ):
                self.finish()
        except asyncio.CancelledError:
            pass
        finally:
            if self.segment_gap_task is asyncio.current_task():
                self.segment_gap_task = None

    async def sync_expression(self) -> None:
        """Bind the model's complete facial timeline to the PCM playback clock."""
        if self.expression_claimed:
            return
        cues = cues_after(self.expression_sequence)
        if not cues:
            return
        self.expression_sequence = cues[-1].sequence
        self.expression_claimed = True
        try:
            for cue in cues:
                await self._client().post(
                    f"{self.base_url}/expression",
                    json={
                        "profile": cue.profile,
                        "intensity": cue.intensity,
                        "duration_ms": cue.duration_ms,
                        "mouth_strength": cue.mouth_strength,
                        "sequence": cue.sequence,
                        "delay_ms": cue.delay_ms,
                    },
                    # Visual direction is optional metadata. It must never hold
                    # up the first spoken PCM packet during renderer recovery.
                    timeout=0.35,
                )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("AVTR-1 expression cue failed: %s", exc)

    def finish(self, *, complete: bool = False) -> None:
        self.done = True
        self.complete_flush = bool(complete)
        if self.started_at is not None and self.finished_at is None:
            self.finished_at = time.monotonic()
        if self.pending and self.pump_task is None:
            self._start_pump()

    def finish_expression_segment(self) -> None:
        self.expression_claimed = False

    async def interrupt(self) -> None:
        self.generation += 1
        self.pending.clear()
        self.done = False
        self.expression_claimed = False
        clear_delivery_state()
        self._reset_metrics()
        task = self.pump_task
        self.pump_task = None
        gap_task = self.segment_gap_task
        self.segment_gap_task = None
        if gap_task is not None:
            gap_task.cancel()
        if task is not None:
            task.cancel()
        try:
            await self._client().post(f"{self.base_url}/interrupt")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("AVTR-1 local interrupt failed: %s", exc)

    def _start_pump(self) -> None:
        generation = self.generation
        # Bind mode/completeness to this exact upload generation. A later
        # response lifecycle event may update the shared tee before this task
        # has drained, but it must not change an in-flight turn's reservoir.
        self.pump_task = asyncio.create_task(
            self._pump(generation, self.playback_mode, self.complete_flush)
        )

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

    async def _pump(
        self, generation: int, playback_mode: str, complete_flush: bool
    ) -> None:
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
                            if complete_flush and self.done
                            else min(PREROLL_BYTES, len(self.pending))
                        )
                        first = False
                    else:
                        size = min(PACE_BYTES, len(self.pending))
                    payload = bytes(self.pending[:size])
                    try:
                        if first is False and self.audio_bytes == len(self.pending):
                            LOG.info(
                                "AVTR playback upload mode=%s complete=%s bytes=%d",
                                playback_mode,
                                complete_flush,
                                len(self.pending),
                            )
                        response = await self._client().post(
                            f"{self.base_url}/audio-chunk",
                            content=payload,
                            params={"mode": playback_mode},
                            headers={"Content-Type": "application/octet-stream"},
                        )
                        response.raise_for_status()
                        # Commit removal only after the gateway acknowledged
                        # this exact packet. Previously it was deleted before
                        # POST, so one transient timeout permanently removed a
                        # sentence from the avatar playback.
                        del self.pending[:size]
                        self.consecutive_post_failures = 0
                    except Exception as exc:  # noqa: BLE001
                        self.consecutive_post_failures += 1
                        # Keep the packet at the head of the queue and retry it.
                        # Log the exception class as timeout messages often have
                        # an empty string representation.
                        if self.consecutive_post_failures == 1 or self.consecutive_post_failures % 10 == 0:
                            LOG.warning(
                                "AVTR-1 local audio tee retry=%d error=%s: %s",
                                self.consecutive_post_failures,
                                type(exc).__name__,
                                exc,
                            )
                        await asyncio.sleep(min(0.5, 0.05 * self.consecutive_post_failures))

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
                            params={
                                **{key: str(value) for key, value in metrics.items()},
                                "mode": playback_mode,
                            },
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
        playback_mode = (
            str(query_params.get("playback_mode", "interactive"))
            if hasattr(query_params, "get")
            else "interactive"
        )
        if is_preview:
            await original_send_events(ws, events)
            return
        connection_id = id(ws)
        if tee.active_connection_id != connection_id:
            # Some realtime backends begin directly with audio deltas and do
            # not emit response.created. Initialize completeness and playback
            # policy from the WebSocket itself so proactive news cannot
            # silently fall back to the interactive 480ms reservoir.
            if complete_audio:
                await tee.interrupt()
            tee.active_connection_id = connection_id
            tee.complete_flush = complete_audio
            tee.playback_mode = (
                "proactive" if playback_mode == "proactive" else "interactive"
            )
            LOG.info(
                "AVTR playback connection mode=%s complete=%s",
                tee.playback_mode,
                complete_audio,
            )
        for event in events:
            event_type = getattr(event, "type", "")
            if event_type == "response.created":
                begin_delivery_response()
                LOG.info(
                    "AVTR playback request mode=%s complete=%s",
                    tee.playback_mode,
                    complete_audio,
                )
            elif event_type in ("response.audio.delta", "response.output_audio.delta"):
                delta = getattr(event, "delta", "")
                if delta:
                    try:
                        await tee.sync_expression()
                        tee.feed(base64.b64decode(delta))
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("Invalid realtime PCM delta: %s", exc)
            elif event_type in ("response.audio.done", "response.output_audio.done"):
                tee.finish_expression_segment()
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
