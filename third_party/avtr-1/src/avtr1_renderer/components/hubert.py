# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""HuBERT speech-feature streaming-chunk policy.

Calls a ``HubertEngine`` and folds in the reference's
``Wav2FeatHubert.__call__`` post-processing: take the last
``2 * n_motion_frames`` HuBERT frames, reshape to
``(B, n_motion_frames, 2, 1024)``, and average each adjacent pair to
downsample 50 Hz -> 25 Hz.

I/O contracts live in ``avtr1_renderer.models.hubert``.
"""

from __future__ import annotations

import torch

from avtr1_renderer.models.hubert import HubertEngine, HubertInput


def run_hubert(
    audio_batch: torch.Tensor,
    *,
    n_motion_frames: int,
    hubert: HubertEngine,
) -> torch.Tensor:
    """Run HuBERT on a batch of tracks and return chunked features.

    HuBERT's effective stride is 320 samples (50 Hz at 16 kHz input). For
    an ``N``-sample input the engine emits ``N // 320`` frames.

    Args:
        audio_batch: ``(B, N)`` float32 CUDA. ``B`` must satisfy the
            engine's optimisation profile.
        n_motion_frames: number of trailing 25 Hz motion frames to
            return per batch element.
        hubert: HuBERT engine (e.g. built via
            ``load_engine(path, HubertInput, HubertOutput)``).

    Returns:
        ``(B, n_motion_frames, 1024)`` float32 CUDA.
    """
    out = hubert(HubertInput(input_values=audio_batch.contiguous()))
    encoding = out.last_hidden_state  # (B, frames_50hz, 1024)
    valid = encoding[:, -n_motion_frames * 2 :]  # (B, 2 * n_motion, 1024)
    B = valid.shape[0]
    return valid.reshape(B, n_motion_frames, 2, 1024).mean(dim=2)
