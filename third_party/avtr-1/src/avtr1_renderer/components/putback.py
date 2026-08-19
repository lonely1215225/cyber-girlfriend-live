# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Composite the rendered face crop back onto the avatar and matte it.

Calls the MODNet matting engine via ``matte_chunk``. The reference's
variant (see ``putback.py:37`` in the old repo) bundles body-motion
simulation, alpha matting, and per-avatar custom mask paths; the
simplified pipeline keeps just the ``warp_affine + mask blend + matte +
bg-composite`` core.

``putback_chunk`` is the only entry point: it processes all N frames at
once in a single batched warp + blend + matting + composite. Single-
frame callers wrap their face in ``unsqueeze(0)`` and read ``[0]`` from
each output -- there is no per-frame variant, because every op in the
pipeline broadcasts cleanly across the leading dim and the engines
support ``B >= 1``.

Conventions:
- Tensors are NCHW float in ``[0, 1]`` on CUDA.
- ``avatar.mask`` is pre-warped to original-frame coordinates at registration
  time, so the inner loop only does a single warp + blend.
- ``avatar.source`` is the grey-bg portrait composite, so MODNet sees a
  clean foreground / background split.
- ``bg`` is passed in by the caller -- the final background the rendered
  head is composited onto using the predicted alpha matte. Backgrounds
  are a render-time choice (the pipeline keeps a registry keyed by id),
  not part of the avatar's identity.
- The output is ``(rgb, alpha)`` -- separate tensors so consumers can
  drop alpha without paying a slice / cat round-trip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from avtr1_renderer.components.matting import matte_chunk

if TYPE_CHECKING:
    from avtr1_renderer.avatar_loader import Avatar
    from avtr1_renderer.models.matting import MODNetEngine


def putback_chunk(
    face_crops: torch.Tensor,
    avatar: Avatar,
    bg: torch.Tensor,
    *,
    matting: MODNetEngine,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched composite: ``(N, 3, H_crop, W_crop)`` faces -> bg-composited frames.

    Args:
        face_crops: (N, 3, H_crop, W_crop) float in [0, 1] -- decoder
            outputs for one chunk.
        avatar: provides ``M_grid`` (precomputed F.affine_grid input),
            ``mask`` (pre-warped pasteback mask), ``source`` (grey-bg
            portrait).
        bg: ``(1, 3, H, W)`` float in ``[0, 1]`` -- the background the
            rendered head is composited over. Broadcasts against the
            ``(N, 3, H, W)`` chunk.
        matting: MODNet alpha-matting engine; called once per chunk.

    Returns:
        ``(rgb, alpha)`` -- ``rgb`` is ``(N, 3, H, W)`` bg-composited
        float in [0, 1]; ``alpha`` is ``(N, 1, H, W)`` MODNet matte in
        [0, 1]. Both on CUDA. Callers that don't need alpha just drop
        it -- no separate code path. We keep them as separate tensors
        so consumers (e.g. ``pack_frames``) don't pay for a concat /
        slice round-trip when they handle the two planes independently
        (the YUV stacked-alpha format does, plain YUV ignores alpha).
    """
    assert face_crops.ndim == 4 and face_crops.shape[1] == 3, (
        f"face_crops must be (N, 3, H, W), got {tuple(face_crops.shape)}"
    )
    n = face_crops.shape[0]
    h, w = avatar.source.shape[-2:]

    # Hand-rolled affine warp: ``F.affine_grid`` builds the sampling
    # grid from the precomputed normalised inverse on ``avatar``, and
    # ``F.grid_sample`` does the bilinear lookup. Two kernels total.
    # ``kornia.warp_affine`` would do the same end-to-end but ~40
    # kernels (it inverts ``M`` on every call via cuBLAS LU, with two
    # D2H sync points to check for singularity); we cache the inverse
    # on Avatar so we never pay that cost at render time.
    M_b = avatar.M_grid.unsqueeze(0).expand(n, -1, -1)
    grid = F.affine_grid(M_b, [n, 3, h, w], align_corners=False)
    face_warped = F.grid_sample(
        face_crops,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    # blended = mask * face + (1 - mask) * source -- one fused kernel via lerp.
    # avatar.mask: (1, H, W); avatar.source: (3, H, W); both broadcast to (N, 3, H, W).
    blended = torch.lerp(avatar.source.unsqueeze(0), face_warped, avatar.mask.unsqueeze(0))
    if avatar.no_matting:
        # Portrait carries its own background — skip MODNet and return as-is.
        alpha = torch.ones(n, 1, h, w, device=blended.device, dtype=blended.dtype)
        return blended, alpha
    alpha = matte_chunk(blended, modnet=matting)  # (N, 1, H, W)
    # composited = alpha * blended + (1 - alpha) * bg -- another fused lerp.
    composited = torch.lerp(bg, blended, alpha)
    return composited, alpha


