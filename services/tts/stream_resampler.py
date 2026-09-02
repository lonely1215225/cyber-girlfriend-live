"""Stateful sample-rate conversion for live TTS patches."""

from __future__ import annotations

import numpy as np

PIPELINE_SR = 16000


class StreamResampler:
    """Keep successive PCM patches continuous instead of resampling each one.

    Independent ``resample_poly`` on each short patch puts a filter transient
    on every boundary and sounds like AM radio flutter.
    """

    def __init__(self, out_rate: int = PIPELINE_SR) -> None:
        self.out_rate = int(out_rate or PIPELINE_SR)
        self._stream = None
        self._in_rate = 0

    def reset(self) -> None:
        self._stream = None
        self._in_rate = 0

    def _ensure(self, sample_rate: int):
        rate = int(sample_rate or self.out_rate)
        if self._stream is not None and self._in_rate == rate:
            return self._stream
        if rate == self.out_rate:
            self._stream = None
            self._in_rate = rate
            return None
        import soxr

        self._stream = soxr.ResampleStream(
            rate, self.out_rate, 1, dtype="float32", quality="HQ"
        )
        self._in_rate = rate
        return self._stream

    def push(
        self, audio: np.ndarray, sample_rate: int, *, last: bool = False
    ) -> np.ndarray:
        pcm = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        stream = self._ensure(sample_rate)
        if stream is None:
            return pcm
        return stream.resample_chunk(pcm, last=last)

    def flush(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        return self._stream.resample_chunk(
            np.zeros(0, dtype=np.float32), last=True
        )


def downsample_to_pipeline(
    audio: np.ndarray, sample_rate: int, target: int = PIPELINE_SR
) -> tuple[np.ndarray, int]:
    if int(sample_rate or target) == int(target):
        return np.ascontiguousarray(audio, dtype=np.float32), int(target)
    import soxr

    return (
        soxr.resample(
            np.ascontiguousarray(audio, dtype=np.float32),
            int(sample_rate),
            int(target),
            quality="HQ",
        ),
        int(target),
    )
