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

import base64
import logging
import os
import sys

import uvicorn

from speech_to_speech.api.openai_realtime import websocket_router

from avatar_tee import LocalAvatarTee
from tiered_memory import cancel_semantic_refinements, install_tiered_memory
from expression_director import begin_delivery_response
from playback_policy import (
    apply_websocket_playback_policy,
    response_is_progress_only,
    should_complete_flush_before_play,
)

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


if TEE_URL:
    tee = LocalAvatarTee(TEE_URL)
    original_send_events = websocket_router._send_events

    async def send_events_with_avatar_tee(ws, events):
        query_params = getattr(ws, "query_params", {})
        is_preview = str(query_params.get("preview", "")) == "1" if hasattr(query_params, "get") else False
        # Room news and welcomes now use the same live socket as chat. The
        # complete_audio flag is only honored when a caller still asks for it.
        complete_audio = (
            str(query_params.get("complete_audio", "")) == "1"
            if hasattr(query_params, "get") else False
        )
        playback_mode = (
            str(query_params.get("playback_mode", "interactive"))
            if hasattr(query_params, "get")
            else "interactive"
        )
        apply_websocket_playback_policy(
            complete_audio=complete_audio,
            playback_mode=playback_mode,
        )
        complete_flush = should_complete_flush_before_play(playback_mode)
        if is_preview:
            await original_send_events(ws, events)
            return
        connection_id = id(ws)
        if tee.active_connection_id != connection_id:
            # Some realtime backends begin directly with audio deltas and do
            # not emit response.created. Initialize completeness and playback
            # policy from the WebSocket itself so proactive news cannot
            # silently fall back to the interactive 480ms reservoir.
            if complete_audio or complete_flush:
                await tee.interrupt()
            tee.active_connection_id = connection_id
            tee.complete_flush = complete_flush
            tee.playback_mode = (
                "proactive" if playback_mode == "proactive" else "interactive"
            )
            LOG.info(
                "AVTR playback connection mode=%s complete=%s flush=%s",
                tee.playback_mode,
                complete_audio,
                complete_flush,
            )
        for event in events:
            event_type = getattr(event, "type", "")
            if event_type == "response.created":
                begin_delivery_response()
                # Wait-beat PCM can go quiet for >8s while the real answer
                # clones. Refresh the segment gap here so the 14s idle close
                # cannot finish the turn mid-sentence and force another 800ms
                # start watermark.
                tee.keep_turn_open()
                LOG.info(
                    "AVTR playback request mode=%s complete=%s flush=%s",
                    tee.playback_mode,
                    complete_audio,
                    complete_flush,
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
                # Do not close the AVTR turn here. The next TTS sentence is
                # still cloning; finishing made the mouth stop and then wait
                # through another full generate plus a new start watermark.
                # response.done (or the long segment gap) closes the turn.
            elif event_type == "input_audio_buffer.speech_started":
                # Semantic memory uses the conversation model only while idle.
                # Cancel it at VAD onset, before STT can submit the next LLM turn.
                cancel_semantic_refinements()
                await tee.interrupt()
            elif event_type == "response.done":
                response = getattr(event, "response", None)
                if getattr(response, "status", "") == "cancelled":
                    await tee.interrupt()
                elif response_is_progress_only(response):
                    # Wait-beat / tool progress is not the end of the spoken
                    # turn. Finishing here froze the face until the real answer
                    # cloned another full sentence. Upstream may drop metadata,
                    # so interactive turns also debounce finish below.
                    tee.keep_turn_open()
                    LOG.info("AVTR keep speech turn open after progress speech")
                elif complete_audio or complete_flush:
                    tee.finish(complete=complete_audio)
                else:
                    tee.schedule_finish(complete=complete_audio)
        await original_send_events(ws, events)

    websocket_router._send_events = send_events_with_avatar_tee
    LOG.info("AVTR-1 local audio tee enabled: %s", TEE_URL)


from speech_to_speech.s2s_pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
