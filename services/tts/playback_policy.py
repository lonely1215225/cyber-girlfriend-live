"""Per-connection TTS policy: batch news, stream dialogue.

The speech-to-speech pipeline has one slot, so a process-wide flag is enough.
News and arrival welcomes already connect with ``complete_audio=1``; those
turns can wait for the full reply, run one VoxCPM generate, then play.
Interactive dialogue must flush the first spoken sentence immediately.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_batch_tts = False


def set_batch_tts(batch: bool) -> None:
    global _batch_tts
    with _lock:
        _batch_tts = bool(batch)


def is_batch_tts() -> bool:
    with _lock:
        return _batch_tts


def should_complete_flush_before_play(playback_mode: str = "interactive") -> bool:
    """News waits for the whole clip. Dialogue and welcomes start after preroll."""

    return str(playback_mode or "") == "proactive"


def apply_websocket_playback_policy(
    *,
    complete_audio: bool = False,
    playback_mode: str = "interactive",
) -> bool:
    """Return True when this connection should synthesize the whole turn at once."""

    batch = bool(complete_audio) or str(playback_mode or "") == "proactive"
    set_batch_tts(batch)
    return batch
