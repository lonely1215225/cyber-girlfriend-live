# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""FastAPI HTTP front-end for ``Pipeline.process_chunk``.

One streaming endpoint plus health. Avatars are loaded at startup by
scanning the ``reference_frames`` artifact directory.

The response body is the new safetensors state blob first, followed by
rendered frames concatenated. ``X-State-Length-Bytes`` gives the state-blob
length so the client can split the body without buffering the whole response.

Audio format: raw int16 PCM at 16 kHz mono (same convention as the old
renderer). The server converts to float32 [-1, 1] internally before
constructing the ``Chunk``.

Run locally::

    AVTR1_LOCAL_STORAGE=/var/lib/avtr1/artifacts \\
    pixi run python -m avtr1_renderer.api.app
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import numpy as np
import uvicorn
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from avtr1_renderer.api.load_balancing import keep_alive_worker
from avtr1_renderer.components.pixel_format import PixelFormat, get_bytes_per_frame
from avtr1_renderer.avtr1_motion_generator import state_from_safetensors, state_to_safetensors
from avtr1_renderer.pipeline import Pipeline, TRANSPARENT_BG_ID

from avtr1_renderer.types import Chunk, RenderOptions
from avtr1_renderer.utils.asyncio import run_in_thread
from avtr1_renderer.utils.cuda_health import CudaHealthChecker

LOG = logging.getLogger(__name__)

_INT16_MAX = 32768.0


def _bytes_to_float32(buf: bytes, expected_samples: int, field: str) -> np.ndarray:
    expected_bytes = expected_samples * 2
    if len(buf) != expected_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field}: expected exactly {expected_bytes} bytes "
                f"({expected_samples} int16 samples), got {len(buf)} bytes"
            ),
        )
    return np.frombuffer(buf, dtype=np.int16).astype(np.float32) / _INT16_MAX


def _build_chunk(
    audio_bytes: tuple[bytes, bytes, bytes, bytes],
    cur_n: int,
    fut_n: int,
) -> Chunk:
    cur, fut, curl, futl = audio_bytes
    speech = np.concatenate([
        _bytes_to_float32(cur, cur_n, "speech_current"),
        _bytes_to_float32(fut, fut_n, "speech_future"),
    ])
    listen = np.concatenate([
        _bytes_to_float32(curl, cur_n, "listen_current"),
        _bytes_to_float32(futl, fut_n, "listen_future"),
    ])
    return Chunk(audio_speech=speech, audio_listen=listen)


def run_chunk(
    loop: asyncio.AbstractEventLoop,
    state_fut: asyncio.Future[bytes],
    send: MemoryObjectSendStream[bytes],
    pipeline: Pipeline,
    avatar,
    audio_bytes: tuple[bytes, bytes, bytes, bytes],
    state_blob_in: bytes | None,
    options: RenderOptions,
    cur_n: int,
    fut_n: int,
) -> None:
    """Drive one chunk end-to-end on a worker thread."""
    try:
        chunk = _build_chunk(audio_bytes, cur_n, fut_n)
        if state_blob_in is not None:
            try:
                prev_state = state_from_safetensors(state_blob_in, device="cuda")
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"failed to decode state safetensors blob: {exc}",
                ) from exc
        else:
            prev_state = None

        next_state, frames_iter = pipeline.process_chunk(avatar, chunk, prev_state, options)
        loop.call_soon_threadsafe(state_fut.set_result, state_to_safetensors(next_state))

        for f in frames_iter:
            loop.call_soon_threadsafe(send.send_nowait, f.data.tobytes())

    except BaseException as exc:
        if not state_fut.done():
            loop.call_soon_threadsafe(state_fut.set_exception, exc)
        else:
            LOG.exception("post-state worker failure; closing connection")
    finally:
        loop.call_soon_threadsafe(send.close)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.monotonic()

    # Auto-discover all portrait PNGs in reference_frames.
    from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager
    mgr = get_artifact_manager()
    portraits_dir = Path(mgr.get_artifact_path("reference_frames"))
    avatar_ids = [p.stem for p in sorted(portraits_dir.glob("*.png"))]
    wanted = os.environ.get("AVTR1_AVATAR_IDS", "").strip()
    if wanted:
        avatar_ids = [x.strip() for x in wanted.split(",") if x.strip()]
    if not avatar_ids:
        raise RuntimeError(f"No avatar PNGs found in {portraits_dir}")

    out_h = int(os.environ.get("AVTR1_OUT_H", "1280"))
    out_w = int(os.environ.get("AVTR1_OUT_W", "720"))
    pipeline, registry = Pipeline.from_artifacts(
        avatar_ids=avatar_ids,
        portraits_dir=portraits_dir,
        out_size=(out_h, out_w),
    )
    LOG.info("Avatars: %s", ", ".join(sorted(registry)))
    LOG.info(
        "Loaded engines + %d avatars + %d backgrounds in %.1fs",
        len(registry),
        len(pipeline._backgrounds),
        time.monotonic() - t0,
    )
    app.state.pipeline = pipeline
    app.state.registry = registry
    app.state.portraits_dir = portraits_dir
    app.state.avatar_loader = pipeline._avatar_loader
    app.state.avatar_load_lock = asyncio.Lock()
    mg = pipeline._motion_generator
    app.state.current_samples = mg.chunk_size * mg.frame_len
    app.state.future_samples = mg.future_size * mg.frame_len + mg.audio_shift
    app.state.health = CudaHealthChecker()
    async with keep_alive_worker():
        yield


app = FastAPI(lifespan=lifespan)


@app.post("/process-audio-v3")
async def process_audio_v3(
    request: Request,
    current_chunk: UploadFile,
    future_chunk: UploadFile,
    current_chunk_listen: UploadFile,
    future_chunk_listen: UploadFile,
    state: UploadFile | None = None,
    avatar_id: str = "anya_03_studio",
    bg_id: str = Query(..., description="Background id; must match an entry from /avatars"),
    pixel_format: PixelFormat = "yuv_i420",
    cfg_self_audio: float = 2.0,
    cfg_other_audio: float = 2.0,
    cfg_kp: float = 4.0,
    noise_alpha: float = 2.0,
    noise_trunc_z: float = 1.2,
) -> StreamingResponse:
    pipeline: Pipeline = request.app.state.pipeline
    registry: dict = request.app.state.registry

    if avatar_id not in registry:
        portraits_dir: Path | None = getattr(request.app.state, "portraits_dir", None)
        loader = getattr(request.app.state, "avatar_loader", None)
        portrait = (portraits_dir / f"{avatar_id}.png") if portraits_dir else None
        if loader is None or portrait is None or not portrait.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Unknown avatar_id {avatar_id!r}; available: {sorted(registry)}",
            )
        async with request.app.state.avatar_load_lock:
            if avatar_id not in registry:
                LOG.info("Loading avatar %s from %s", avatar_id, portrait)
                registry[avatar_id] = await run_in_thread(
                    lambda: loader.load(portrait, avatar_id=avatar_id)
                )

    avatar = registry[avatar_id]
    if bg_id.endswith(".png"):
        bg_id = bg_id[:-4]

    audio_bytes = (
        await current_chunk.read(),
        await future_chunk.read(),
        await current_chunk_listen.read(),
        await future_chunk_listen.read(),
    )

    state_blob_in = await state.read() if state is not None else None
    if state_blob_in is not None and len(state_blob_in) == 0:
        state_blob_in = None

    options = RenderOptions(
        pixel_format=pixel_format,
        bg_id=bg_id,
        cfg_self_audio=cfg_self_audio,
        cfg_other_audio=cfg_other_audio,
        cfg_kp=cfg_kp,
        noise_alpha=noise_alpha,
        noise_trunc_z=noise_trunc_z,
    )

    loop = asyncio.get_running_loop()
    state_fut: asyncio.Future[bytes] = loop.create_future()
    send, recv = anyio.create_memory_object_stream[bytes](max_buffer_size=math.inf)

    run_in_thread(
        run_chunk,
        loop,
        state_fut,
        send,
        pipeline,
        avatar,
        audio_bytes,
        state_blob_in,
        options,
        request.app.state.current_samples,
        request.app.state.future_samples,
    )

    state_blob = await state_fut

    bg = pipeline._backgrounds[options.bg_id]
    out_h, out_w = int(bg.shape[-2]), int(bg.shape[-1])
    frame_bytes = get_bytes_per_frame(out_h, out_w, options.pixel_format)
    n_frames = pipeline._motion_generator.chunk_size

    async def body():
        try:
            yield state_blob
            async with recv:
                async for frame_chunk in recv:
                    yield frame_chunk
        except BaseException:
            logging.warning("Body streaming ended abruptly", exc_info=True)

    return StreamingResponse(
        body(),
        headers={
            "X-Num-Frames": str(n_frames),
            "X-Frame-Height": str(out_h),
            "X-Frame-Width": str(out_w),
            "X-Frame-Length-Bytes": str(frame_bytes),
            "X-Has-State": "yes",
            "X-State-Format": "safetensors",
            "X-State-Length-Bytes": str(len(state_blob)),
        },
    )


@app.get("/avatars")
async def avatars(request: Request) -> dict:
    """List the avatar ids and background ids loaded at startup (sorted).

    Backgrounds excludes the reserved ``transparent`` sentinel since callers
    pick it via ``pixel_format`` rather than as a real background image.
    """
    portraits_dir: Path | None = getattr(request.app.state, "portraits_dir", None)
    disk: list[str] = []
    if portraits_dir is not None and portraits_dir.is_dir():
        disk = [p.stem for p in sorted(portraits_dir.glob("*.png"))]
    registry: dict = getattr(request.app.state, "registry", None) or {}
    pipeline: Pipeline | None = getattr(request.app.state, "pipeline", None)
    backgrounds: list[str] = []
    if pipeline is not None:
        backgrounds = sorted(b for b in pipeline._backgrounds if b != TRANSPARENT_BG_ID)
    return {
        "avatars": disk or sorted(registry),
        "loaded": sorted(registry),
        "backgrounds": backgrounds,
    }


@app.get("/health")
async def health(request: Request):
    checker: CudaHealthChecker | None = getattr(request.app.state, "health", None)
    if checker is None:
        raise HTTPException(status_code=503, detail="starting")
    try:
        await checker.check()
    except Exception as exc:
        LOG.exception("cuda healthcheck failed")
        raise HTTPException(status_code=503, detail=f"cuda unhealthy: {exc}") from exc
    return {"status": "ok"}


class _DropSuccessfulAccess(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        status = record.args[4] if isinstance(record.args, tuple) and len(record.args) >= 5 else None
        if isinstance(status, int) and 200 <= status < 400:
            return False
        return True


# Applied at import time so it sticks whether this module is launched via
# `python -m avtr1_renderer.api.app` or via `python -m uvicorn avtr1_renderer.api.app:app`
# (the orchestrator uses the latter to pass --port). Uvicorn configures its own
# logging before importing the app, so our filter attaches after its handlers exist.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").addFilter(_DropSuccessfulAccess())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run("avtr1_renderer.api.app:app", host="0.0.0.0", port=8000)
