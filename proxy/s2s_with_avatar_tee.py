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

import httpx
import uvicorn

from speech_to_speech.api.openai_realtime import websocket_router

from tiered_memory import cancel_semantic_refinements, install_tiered_memory

LOG = logging.getLogger("speech_to_speech.avatar_tee")
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
PREROLL_MS = max(200, int(os.environ.get("AVATAR_TEE_PREROLL_MS", "480")))
PREROLL_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * PREROLL_MS / 1000)
PACE_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.1)


class LocalAvatarTee:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.pending = bytearray()
        self.generation = 0
        self.done = False
        self.pump_task: asyncio.Task | None = None
        self.client: httpx.AsyncClient | None = None

    def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=httpx.Timeout(2.0))
        return self.client

    def feed(self, pcm: bytes) -> None:
        if not pcm:
            return
        self.pending.extend(pcm)
        if self.pump_task is None and len(self.pending) >= PREROLL_BYTES:
            self._start_pump()

    def finish(self) -> None:
        self.done = True
        if self.pending and self.pump_task is None:
            self._start_pump()

    async def interrupt(self) -> None:
        self.generation += 1
        self.pending.clear()
        self.done = False
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
                        size = min(PREROLL_BYTES, len(self.pending))
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
                    try:
                        await self._client().post(f"{self.base_url}/audio-finish")
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("AVTR-1 local audio finish failed: %s", exc)
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
        if is_preview:
            await original_send_events(ws, events)
            return
        for event in events:
            event_type = getattr(event, "type", "")
            if event_type in ("response.audio.delta", "response.output_audio.delta"):
                delta = getattr(event, "delta", "")
                if delta:
                    try:
                        tee.feed(base64.b64decode(delta))
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("Invalid realtime PCM delta: %s", exc)
            elif event_type in ("response.audio.done", "response.output_audio.done"):
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
                    tee.finish()
        await original_send_events(ws, events)

    websocket_router._send_events = send_events_with_avatar_tee
    LOG.info("AVTR-1 local audio tee enabled: %s", TEE_URL)


from speech_to_speech.s2s_pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
