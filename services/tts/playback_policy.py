"""Per-connection TTS policy: live clone by default.

The speech-to-speech pipeline has one slot, so a process-wide flag is enough.
Chat, arrival welcomes, and room news all start the first sentence after the
short live TTS reservoir. ``complete_audio=1`` remains available if a caller
still wants one whole-clip generate.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_batch_tts = False
_live_sentence_index = 0


def set_batch_tts(batch: bool) -> None:
    global _batch_tts
    with _lock:
        _batch_tts = bool(batch)


def begin_live_tts_turn() -> None:
    """First sentence of a new reply keeps the start reservoir."""

    global _live_sentence_index
    with _lock:
        _live_sentence_index = 0


def take_live_tts_sentence() -> int:
    """Return the 0-based live sentence index, then advance it."""

    global _live_sentence_index
    with _lock:
        index = _live_sentence_index
        _live_sentence_index += 1
        return index


def is_batch_tts() -> bool:
    with _lock:
        return _batch_tts


def response_is_progress_only(response: object | None) -> bool:
    """True for the isolated wait-beat / tool-progress readout."""
    if response is None:
        return False
    metadata = getattr(response, "metadata", None)
    if isinstance(metadata, dict):
        purpose = metadata.get("client_purpose")
    else:
        purpose = getattr(metadata, "client_purpose", "") if metadata is not None else ""
    return str(purpose or "") == "tool_progress"


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
