#!/usr/bin/env python3
"""Run speech-to-speech with a server-local PCM tee for AVTR-1.

The realtime API normally sends generated PCM to the browser.  Sending that
PCM back over public HTTP to animate AVTR-1 makes lip sync depend on a second
network round trip.  This wrapper mirrors the same audio events to the local
avatar gateway, with a short preroll and realtime pacing, before starting the
normal CLI.
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

LOG = logging.getLogger("speech_to_speech.avatar_tee")
# The local pacer posts ten small chunks per second; per-request INFO logs would
# drown out the useful latency and pipeline messages.
logging.getLogger("httpx").setLevel(logging.WARNING)
_uvicorn_config_init = uvicorn.Config.__init__


def _quiet_uvicorn_config(self, *args, **kwargs):
    kwargs.setdefault("access_log", False)
    _uvicorn_config_init(self, *args, **kwargs)


uvicorn.Config.__init__ = _quiet_uvicorn_config
TEE_URL = os.environ.get("AVTR1_LOCAL_TEE_URL", "").rstrip("/")
SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
PREROLL_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.6)
PACE_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.1)
PACE_SECONDS = 0.1


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
        deadline = asyncio.get_running_loop().time()
        try:
            while generation == self.generation:
                threshold = PREROLL_BYTES if first else PACE_BYTES
                if len(self.pending) >= threshold or (self.done and self.pending):
                    if first:
                        size = len(self.pending)
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

                    deadline += PACE_SECONDS
                    await asyncio.sleep(max(0.0, deadline - asyncio.get_running_loop().time()))
                    continue

                if self.done and not self.pending:
                    self.done = False
                    return
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        finally:
            if generation == self.generation and self.pump_task is asyncio.current_task():
                self.pump_task = None


if TEE_URL:
    tee = LocalAvatarTee(TEE_URL)
    original_send_events = websocket_router._send_events

    async def send_events_with_avatar_tee(ws, events):
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
