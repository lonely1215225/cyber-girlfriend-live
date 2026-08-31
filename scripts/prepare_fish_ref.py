#!/usr/bin/env python3
"""Stretch a short clone clip by appending a tail of itself."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--tail-seconds", type=float, default=6.0)
    args = parser.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    with wave.open(str(src), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
        rate = reader.getframerate()
        width = reader.getsampwidth() * reader.getnchannels()
    total_seconds = len(frames) / width / rate
    tail_seconds = min(max(1.0, args.tail_seconds), max(1.0, total_seconds - 0.4))
    tail_bytes = int(tail_seconds * rate) * width
    extended = frames + frames[-tail_bytes:]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(extended)
    print(
        f"wrote {dst} {total_seconds:.2f}s + {tail_seconds:.2f}s -> "
        f"{len(extended) / width / rate:.2f}s"
    )


if __name__ == "__main__":
    main()
