#!/usr/bin/env python3
"""AVTR-1 sidecar that publishes synchronized HTTP-FLV for the browser."""
from __future__ import annotations

import asyncio
import io
import os
import time
import wave
from fractions import Fraction

import aiohttp
import av
import numpy as np
from aiohttp import web

RENDERER = os.environ.get("AVTR1_URL", "http://127.0.0.1:18012").rstrip("/")
HOST = os.environ.get("AVATAR_GW_HOST", "127.0.0.1")
PORT = int(os.environ.get("AVATAR_GW_PORT", "18011"))
AVATAR_ID = os.environ.get("AVTR1_AVATAR_ID", "xiaoya")
BG_ID = os.environ.get("AVTR1_BG_ID", "plain_white")
H264_BITRATE = int(os.environ.get("AVTR1_H264_BITRATE", "1800000"))
CFG_SELF_AUDIO = float(os.environ.get("AVTR1_CFG_SELF_AUDIO", "2.3"))
CFG_OTHER_AUDIO = float(os.environ.get("AVTR1_CFG_OTHER_AUDIO", "2.0"))
CFG_KP = float(os.environ.get("AVTR1_CFG_KP", "3.0"))
NOISE_ALPHA = float(os.environ.get("AVTR1_NOISE_ALPHA", "1.5"))
NOISE_TRUNC_Z = float(os.environ.get("AVTR1_NOISE_TRUNC_Z", "1.0"))

SAMPLE_RATE = 16_000
CHUNK_SIZE = 5
FRAME_LEN = 640
AUDIO_SHIFT = 80
CURRENT_SAMPLES = CHUNK_SIZE * FRAME_LEN
FUTURE_SAMPLES = CHUNK_SIZE * FRAME_LEN + AUDIO_SHIFT
WINDOW_SAMPLES = CURRENT_SAMPLES + FUTURE_SAMPLES
PCM_PACKET_BYTES = 640
MAX_SPEECH_BYTES = SAMPLE_RATE * 2 * 30

last_frame_at = 0.0
connected = False
state_blob: bytes | None = None
state_avatar_id: str | None = None
speech_pcm = bytearray()
listen_pcm = bytearray()
buf_lock = asyncio.Lock()
flv_subscribers: set[asyncio.Queue] = set()
video_pace_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
audio_pace_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
h264_encoder: H264Encoder | None = None
h264_bytes = 0
renderer_session: aiohttp.ClientSession | None = None
flv_muxer: FlvMuxer | None = None


class H264Encoder:
    """Persistent x264 encoder: one Annex-B access unit per input frame."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.pts = 0
        ctx = av.CodecContext.create("libx264", "w")
        ctx.width = width
        ctx.height = height
        ctx.pix_fmt = "yuv420p"
        ctx.time_base = Fraction(1, 25)
        ctx.framerate = Fraction(25, 1)
        ctx.bit_rate = H264_BITRATE
        ctx.gop_size = 12
        ctx.max_b_frames = 0
        ctx.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "x264-params": "repeat-headers=1:scenecut=0:keyint=12:min-keyint=12:bframes=0:rc-lookahead=0:sync-lookahead=0",
        }
        ctx.open()
        self.ctx = ctx

    def encode(self, raw_i420: bytes) -> list[tuple[bytes, bool]]:
        yuv = np.frombuffer(raw_i420, dtype=np.uint8).reshape(
            (self.height * 3 // 2, self.width)
        )
        frame = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
        frame.pts = self.pts
        self.pts += 1
        return [(bytes(packet), bool(packet.is_keyframe)) for packet in self.ctx.encode(frame)]


def _split_annexb(data: bytes) -> list[bytes]:
    nalus: list[bytes] = []
    i = 0
    n = len(data)
    while i < n:
        if i + 3 < n and data[i : i + 4] == b"\x00\x00\x00\x01":
            i += 4
        elif i + 2 < n and data[i : i + 3] == b"\x00\x00\x01":
            i += 3
        else:
            i += 1
            continue
        start = i
        while i < n:
            if i + 3 < n and data[i : i + 4] == b"\x00\x00\x00\x01":
                break
            if i + 2 < n and data[i : i + 3] == b"\x00\x00\x01":
                break
            i += 1
        if i > start:
            nalus.append(data[start:i])
    return nalus


def _avcc_payload(nalus: list[bytes]) -> bytes:
    chunks: list[bytes] = []
    for nalu in nalus:
        chunks.append(len(nalu).to_bytes(4, "big"))
        chunks.append(nalu)
    return b"".join(chunks)


class FlvMuxer:
    HEADER = b"FLV\x01\x05\x00\x00\x00\x09\x00\x00\x00\x00"

    def __init__(self) -> None:
        self.avc_header: bytes | None = None
        self.aac_header: bytes | None = None
        self.timestamp_ms = 0
        self.pcm_buf = bytearray()
        self._aac: av.CodecContext | None = None

    def _tag(self, tag_type: int, payload: bytes, ts: int) -> bytes:
        data_size = len(payload)
        header = bytes(
            (
                tag_type,
                (data_size >> 16) & 0xFF,
                (data_size >> 8) & 0xFF,
                data_size & 0xFF,
                (ts >> 16) & 0xFF,
                (ts >> 8) & 0xFF,
                ts & 0xFF,
                (ts >> 24) & 0xFF,
                0,
                0,
                0,
            )
        )
        return header + payload + (11 + data_size).to_bytes(4, "big")

    def _ensure_aac(self) -> av.CodecContext:
        if self._aac is not None:
            return self._aac
        ctx = av.CodecContext.create("aac", "w")
        ctx.sample_rate = SAMPLE_RATE
        ctx.layout = "mono"
        ctx.format = "fltp"
        ctx.bit_rate = 64_000
        ctx.open()
        extra = bytes(ctx.extradata or b"") or bytes((0x14, 0x08))
        self._aac = ctx
        self.aac_header = self._tag(8, bytes((0xAE, 0x00)) + extra, 0)
        return ctx

    def bootstrap(self) -> bytes:
        chunks = [self.HEADER]
        if self.avc_header:
            chunks.append(self.avc_header)
        if self.aac_header:
            chunks.append(self.aac_header)
        return b"".join(chunks)

    def video_tags(self, annexb: bytes, keyframe: bool) -> list[bytes]:
        nalus = _split_annexb(annexb)
        if not nalus:
            return []
        sps = next((n for n in nalus if n and n[0] & 0x1F == 7), None)
        pps = next((n for n in nalus if n and n[0] & 0x1F == 8), None)
        tags: list[bytes] = []
        if sps and pps:
            record = bytes((0x01, sps[1], sps[2], sps[3], 0xFF, 0xE1))
            record += len(sps).to_bytes(2, "big") + sps + bytes((0x01,))
            record += len(pps).to_bytes(2, "big") + pps
            header = self._tag(9, bytes((0x17, 0x00, 0x00, 0x00, 0x00)) + record, self.timestamp_ms)
            self.avc_header = header
            tags.append(header)
        framed = [n for n in nalus if n and (n[0] & 0x1F) not in (7, 8, 9)]
        if not framed:
            return tags
        frame_type = 0x17 if keyframe else 0x27
        tags.append(
            self._tag(
                9,
                bytes((frame_type, 0x01, 0x00, 0x00, 0x00)) + _avcc_payload(framed),
                self.timestamp_ms,
            )
        )
        return tags

    def audio_tags(self, pcm: bytes) -> list[bytes]:
        self._ensure_aac()
        self.pcm_buf.extend(pcm)
        frame_size = int(self._aac.frame_size or 1024)
        tags: list[bytes] = []
        while len(self.pcm_buf) >= frame_size * 2:
            chunk = bytes(self.pcm_buf[: frame_size * 2])
            del self.pcm_buf[: frame_size * 2]
            pcm16 = np.frombuffer(chunk, dtype=np.int16)
            flt = (pcm16.astype(np.float32) / 32768.0).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(flt, format="fltp", layout="mono")
            frame.sample_rate = SAMPLE_RATE
            for packet in self._aac.encode(frame):
                tags.append(self._tag(8, bytes((0xAE, 0x01)) + bytes(packet), self.timestamp_ms))
        return tags

    def advance(self, ms: int = 40) -> None:
        self.timestamp_ms += ms


def publish_flv(data: bytes) -> None:
    if not data:
        return
    for q in tuple(flv_subscribers):
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


def enqueue_paced(q: asyncio.Queue, data) -> None:
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        q.put_nowait(data)
    except asyncio.QueueFull:
        pass


def wav_to_pcm16(raw: bytes) -> bytes:
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        with wave.open(io.BytesIO(raw), "rb") as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if sw != 2:
            raise ValueError(f"unsupported sample width {sw}")
        pcm = np.frombuffer(frames, dtype=np.int16)
        if nch > 1:
            pcm = pcm.reshape(-1, nch).mean(axis=1).astype(np.int16)
        if rate != SAMPLE_RATE:
            x = np.linspace(0, 1, num=len(pcm), endpoint=False)
            n_out = int(round(len(pcm) * SAMPLE_RATE / rate))
            xp = np.linspace(0, 1, num=n_out, endpoint=False)
            pcm = np.interp(xp, x, pcm.astype(np.float32)).astype(np.int16)
        return pcm.tobytes()
    return raw


async def pace_av() -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time()
    last_audio = b"\0" * PCM_PACKET_BYTES
    while True:
        try:
            fresh_video = video_pace_queue.get_nowait()
        except asyncio.QueueEmpty:
            fresh_video = None
        if fresh_video:
            packets = fresh_video
            if flv_muxer is not None:
                for packet, keyframe in packets:
                    for tag in flv_muxer.video_tags(packet, keyframe):
                        publish_flv(tag)
        for _ in range(2):
            try:
                last_audio = audio_pace_queue.get_nowait()
            except asyncio.QueueEmpty:
                last_audio = b"\0" * len(last_audio)
            if flv_muxer is not None:
                for tag in flv_muxer.audio_tags(last_audio):
                    publish_flv(tag)
        if flv_muxer is not None:
            flv_muxer.advance(40)
        deadline += 0.04
        await asyncio.sleep(max(0.0, deadline - loop.time()))


def _window_from_speech(buf: bytearray) -> tuple[bytes, bytes, bytes]:
    need = WINDOW_SAMPLES * 2
    if len(buf) >= need:
        window = bytes(buf[:need])
        del buf[: CURRENT_SAMPLES * 2]
    else:
        window = bytes(buf) + bytes(need - len(buf))
        buf.clear()
    cur = window[: CURRENT_SAMPLES * 2]
    fut = window[CURRENT_SAMPLES * 2 :]
    return cur, fut, cur


def _window_from_listen(buf: bytearray) -> tuple[bytes, bytes]:
    """Return a look-ahead listener window without consuming partial audio.

    AVTR-1 needs about 405ms of current+future context. Holding the first
    partial window avoids alternating real microphone audio and padded silence,
    which otherwise makes the listening motion visibly twitch.
    """
    need = WINDOW_SAMPLES * 2
    if len(buf) < need:
        return bytes(CURRENT_SAMPLES * 2), bytes(FUTURE_SAMPLES * 2)
    window = bytes(buf[:need])
    del buf[: CURRENT_SAMPLES * 2]
    return window[: CURRENT_SAMPLES * 2], window[CURRENT_SAMPLES * 2 :]


async def render_loop() -> None:
    global last_frame_at, connected, state_blob, state_avatar_id, h264_encoder, h264_bytes, renderer_session
    loop = asyncio.get_running_loop()
    while True:
        t0 = loop.time()
        try:
            async with buf_lock:
                cur, fut, played = _window_from_speech(speech_pcm)
                listen_cur, listen_fut = _window_from_listen(listen_pcm)
                avatar_id = AVATAR_ID
                blob = state_blob if state_avatar_id == avatar_id else None
            form = aiohttp.FormData()
            form.add_field("current_chunk", cur, filename="cur.raw", content_type="application/octet-stream")
            form.add_field("future_chunk", fut, filename="fut.raw", content_type="application/octet-stream")
            form.add_field(
                "current_chunk_listen",
                listen_cur,
                filename="curl.raw",
                content_type="application/octet-stream",
            )
            form.add_field(
                "future_chunk_listen",
                listen_fut,
                filename="futl.raw",
                content_type="application/octet-stream",
            )
            if blob:
                form.add_field("state", blob, filename="state.bin", content_type="application/octet-stream")
            params = {
                "avatar_id": avatar_id,
                "bg_id": BG_ID,
                "pixel_format": "yuv_i420",
                "cfg_self_audio": str(CFG_SELF_AUDIO),
                "cfg_other_audio": str(CFG_OTHER_AUDIO),
                "cfg_kp": str(CFG_KP),
                "noise_alpha": str(NOISE_ALPHA),
                "noise_trunc_z": str(NOISE_TRUNC_Z),
            }
            if renderer_session is None:
                raise RuntimeError("renderer HTTP session is not initialized")
            async with renderer_session.post(f"{RENDERER}/process-audio-v3", data=form, params=params) as r:
                if r.status != 200:
                    body = await r.text()
                    print(f"[avtr1-gw] renderer {r.status}: {body[:300]}", flush=True)
                    connected = False
                    async with buf_lock:
                        if state_avatar_id == avatar_id:
                            state_blob = None
                            state_avatar_id = None
                    await asyncio.sleep(0.4)
                    continue
                state_len = int(r.headers["X-State-Length-Bytes"])
                h = int(r.headers["X-Frame-Height"])
                w = int(r.headers["X-Frame-Width"])
                frame_len = int(r.headers["X-Frame-Length-Bytes"])
                n_frames = int(r.headers["X-Num-Frames"])
                body = await r.read()
            next_state = body[:state_len]
            frames = body[state_len:]
            async with buf_lock:
                if AVATAR_ID == avatar_id:
                    state_blob = next_state
                    state_avatar_id = avatar_id
            for i in range(n_frames):
                raw = frames[i * frame_len : (i + 1) * frame_len]
                if len(raw) != frame_len:
                    break
                if h264_encoder is None or (h264_encoder.width, h264_encoder.height) != (w, h):
                    h264_encoder = H264Encoder(w, h)
                    print(f"[avtr1-gw] H.264 {w}x{h} 25fps bitrate={H264_BITRATE}", flush=True)
                packets = h264_encoder.encode(raw)
                h264_bytes += sum(len(packet) for packet, _ in packets)
                last_frame_at = time.time()
                connected = True
                enqueue_paced(video_pace_queue, packets)
                off = i * 2 * PCM_PACKET_BYTES
                enqueue_paced(audio_pace_queue, played[off : off + PCM_PACKET_BYTES] or bytes(PCM_PACKET_BYTES))
                enqueue_paced(
                    audio_pace_queue,
                    played[off + PCM_PACKET_BYTES : off + 2 * PCM_PACKET_BYTES] or bytes(PCM_PACKET_BYTES),
                )
            elapsed = loop.time() - t0
            await asyncio.sleep(max(0.0, 0.2 - elapsed))
        except Exception as exc:
            connected = False
            async with buf_lock:
                state_blob = None
                state_avatar_id = None
            print("[avtr1-gw] render failed:", exc, flush=True)
            await asyncio.sleep(0.5)


async def append_speech(pcm: bytes) -> None:
    if not pcm:
        return
    async with buf_lock:
        speech_pcm.extend(pcm)
        overflow = len(speech_pcm) - MAX_SPEECH_BYTES
        if overflow > 0:
            del speech_pcm[: overflow - (overflow % 2)]


async def append_listen(pcm: bytes) -> None:
    if not pcm:
        return
    async with buf_lock:
        listen_pcm.extend(pcm)
        overflow = len(listen_pcm) - MAX_SPEECH_BYTES
        if overflow > 0:
            del listen_pcm[: overflow - (overflow % 2)]


AVATAR_LABELS = {
    "xiaoya": "小雅",
    "xiaoya_beach_close": "海边近景",
    "xiaoya_beach": "海边",
    "xiaoya_locket": "白背心",
}


def _avatar_label(avatar_id: str) -> str:
    return AVATAR_LABELS.get(avatar_id, avatar_id.replace("_", " ").title())


async def _list_avatar_ids() -> tuple[list[str], list[str]]:
    ids: list[str] = []
    loaded: list[str] = []
    if renderer_session is not None:
        try:
            async with renderer_session.get(f"{RENDERER}/avatars") as response:
                if response.status == 200:
                    payload = await response.json()
                    ids = list(payload.get("avatars") or [])
                    loaded = list(payload.get("loaded") or [])
        except Exception:
            pass
    if not ids:
        ids = [AVATAR_ID]
    preferred = ["xiaoya", "xiaoya_beach_close", "xiaoya_beach", "xiaoya_locket"]
    ordered = [item for item in preferred if item in ids]
    return ordered or [AVATAR_ID], loaded


async def handle_status(_request):
    return web.json_response(
        {
            "connected": connected and last_frame_at > 0,
            "backend": "avtr1",
            "avatar_id": AVATAR_ID,
            "age_ms": int((time.time() - last_frame_at) * 1000) if last_frame_at else None,
            "speech_ms": int(len(speech_pcm) / 2 / SAMPLE_RATE * 1000),
            "listen_ms": int(len(listen_pcm) / 2 / SAMPLE_RATE * 1000),
            "flv_clients": len(flv_subscribers),
            "h264_bitrate": H264_BITRATE,
            "h264_bytes": h264_bytes,
        }
    )


async def handle_avatars(_request):
    ids, loaded = await _list_avatar_ids()
    return web.json_response(
        {
            "avatar_id": AVATAR_ID,
            "avatars": [
                {
                    "id": item,
                    "label": _avatar_label(item),
                    "preview": f"/avatar/avatars/{item}.jpg",
                    "loaded": item in loaded,
                }
                for item in ids
            ],
        }
    )


async def handle_set_avatar(request):
    global AVATAR_ID, state_blob, state_avatar_id, h264_encoder
    body = await request.json()
    avatar_id = str(body.get("avatar_id") or "").strip()
    if not avatar_id or "/" in avatar_id or ".." in avatar_id:
        return web.json_response({"ok": False, "error": "invalid avatar_id"}, status=400)
    ids, _loaded = await _list_avatar_ids()
    if avatar_id not in ids:
        return web.json_response({"ok": False, "error": "unknown avatar"}, status=404)
    async with buf_lock:
        AVATAR_ID = avatar_id
        speech_pcm.clear()
        listen_pcm.clear()
        state_blob = None
        state_avatar_id = None
        h264_encoder = None
    for q in (video_pace_queue, audio_pace_queue):
        while not q.empty():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break
    print(f"[avtr1-gw] avatar -> {AVATAR_ID}", flush=True)
    return web.json_response({"ok": True, "avatar_id": AVATAR_ID, "label": _avatar_label(AVATAR_ID)})


async def handle_livestream(request):
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "video/x-flv",
            "Cache-Control": "no-cache, no-store",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    if flv_muxer is not None:
        bootstrap = flv_muxer.bootstrap()
        if bootstrap:
            try:
                await resp.write(bootstrap)
                await resp.drain()
            except (ConnectionResetError, ConnectionError, asyncio.CancelledError):
                return resp
    flv_subscribers.add(q)
    try:
        while True:
            chunk = await q.get()
            await resp.write(chunk)
            await resp.drain()
    except (ConnectionResetError, ConnectionError, asyncio.CancelledError, RuntimeError):
        pass
    finally:
        flv_subscribers.discard(q)
    return resp


async def handle_audio(request):
    if request.content_type and "multipart" in request.content_type:
        data = await request.post()
        fileobj = data.get("file")
        raw = fileobj.file.read() if fileobj is not None else await request.read()
    else:
        raw = await request.read()
    if not raw:
        return web.json_response({"ok": False, "error": "empty"}, status=400)
    try:
        pcm = wav_to_pcm16(raw)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    await append_speech(pcm)
    return web.json_response({"ok": True, "samples": len(pcm) // 2})


async def handle_audio_chunk(request):
    raw = await request.read()
    await append_speech(raw)
    return web.json_response({"ok": True, "bytes": len(raw)})


async def handle_listen_chunk(request):
    raw = await request.read()
    await append_listen(raw)
    return web.json_response({"ok": True, "bytes": len(raw)})


async def handle_listen_reset(_request):
    async with buf_lock:
        listen_pcm.clear()
    return web.json_response({"ok": True})


async def handle_interrupt(_request):
    global state_blob
    async with buf_lock:
        speech_pcm.clear()
        state_blob = None
    for q in (video_pace_queue, audio_pace_queue):
        while not q.empty():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break
    return web.json_response({"ok": True})


async def on_startup(app):
    global renderer_session, flv_muxer
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=5, sock_read=30)
    renderer_session = aiohttp.ClientSession(timeout=timeout)
    flv_muxer = FlvMuxer()
    app["pacer"] = asyncio.create_task(pace_av())
    app["render"] = asyncio.create_task(render_loop())


async def on_cleanup(app):
    global renderer_session
    for key in ("pacer", "render"):
        app[key].cancel()
    if renderer_session is not None:
        await renderer_session.close()
        renderer_session = None


def main():
    app = web.Application(client_max_size=80 * 1024 * 1024)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/avatars", handle_avatars)
    app.router.add_post("/avatar", handle_set_avatar)
    app.router.add_get("/livestream.flv", handle_livestream)
    app.router.add_post("/audio", handle_audio)
    app.router.add_post("/audio-chunk", handle_audio_chunk)
    app.router.add_post("/listen-chunk", handle_listen_chunk)
    app.router.add_post("/listen-reset", handle_listen_reset)
    app.router.add_post("/interrupt", handle_interrupt)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    print(f"avtr1 gateway {HOST}:{PORT} -> {RENDERER} avatar={AVATAR_ID} bg={BG_ID}", flush=True)
    web.run_app(app, host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
