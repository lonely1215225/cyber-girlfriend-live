"""Server-local PCM tee that keeps one AVTR speech turn across TTS clauses."""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx

from expression_director import clear_delivery_state, cues_after

LOG = logging.getLogger("speech_to_speech.avatar_tee")

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
PREROLL_MS = max(240, int(os.environ.get("AVATAR_TEE_UPLOAD_PREROLL_MS", "320")))
PREROLL_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * PREROLL_MS / 1000)
PACE_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.1)
# The next TTS sentence can take several seconds. The old 1.2s gap closed the mouth
# and forced another AVTR start watermark after every clause.
SEGMENT_GAP_SECONDS = max(
    8.0, float(os.environ.get("AVATAR_TEE_SEGMENT_GAP_MS", "14000")) / 1000.0
)
# Interactive response.done used to call finish() immediately. That closed the
# AVTR turn after a wait-beat or the first clause, then froze the face until
# the next IndexTTS sentence filled another 800ms reservoir. Hold the turn
# open long enough for the next clone to arrive.
FINISH_HOLD_SECONDS = max(
    4.0, float(os.environ.get("AVATAR_TEE_FINISH_HOLD_MS", "8000")) / 1000.0
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
        self._finish_task: asyncio.Task | None = None
        self._finish_complete = False

    def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=httpx.Timeout(2.0))
        return self.client

    def feed(self, pcm: bytes) -> None:
        if not pcm:
            return
        now = time.monotonic()
        if self.done:
            # The 14s segment-gap finish can fire while idle. New PCM must
            # reopen the turn; otherwise the pump drains one preroll packet
            # and the gateway restarts H.264 on a 300ms clip.
            self.done = False
            self.finished_at = None
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
        self._cancel_scheduled_finish()
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
        self, generation: int, marker: float | None, delay: float | None = None
    ) -> None:
        """Close a spoken segment while a slow cloud continuation is pending."""
        try:
            await asyncio.sleep(SEGMENT_GAP_SECONDS if delay is None else delay)
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

    def keep_turn_open(self, *, delay: float | None = None) -> None:
        """Hold the AVTR turn while the next response is still cloning."""
        self._cancel_scheduled_finish()
        if self.started_at is None and self.last_feed_at is None:
            return
        if self.segment_gap_task is not None:
            self.segment_gap_task.cancel()
        marker = time.monotonic()
        self.last_feed_at = marker
        hold = SEGMENT_GAP_SECONDS if delay is None else max(1.0, float(delay))
        self.segment_gap_task = asyncio.create_task(
            self._finish_segment_after_gap(self.generation, marker, hold)
        )

    def _cancel_scheduled_finish(self) -> None:
        task = self._finish_task
        self._finish_task = None
        if task is not None and not task.done():
            task.cancel()

    def schedule_finish(self, *, complete: bool, delay: float | None = None) -> None:
        """Close the AVTR turn after a pause, unless more PCM arrives first."""
        self._cancel_scheduled_finish()
        self._finish_complete = bool(complete)
        hold = FINISH_HOLD_SECONDS if delay is None else max(0.0, float(delay))
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.finish(complete=complete)
            return
        self._finish_task = loop.create_task(self._delayed_finish(hold))

    async def _delayed_finish(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        self._finish_task = None
        self.finish(complete=self._finish_complete)

    def finish(self, *, complete: bool = False) -> None:
        self._cancel_scheduled_finish()
        self.done = True
        self.complete_flush = bool(complete)
        gap_task = self.segment_gap_task
        self.segment_gap_task = None
        if gap_task is not None:
            gap_task.cancel()
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
        self._cancel_scheduled_finish()
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
