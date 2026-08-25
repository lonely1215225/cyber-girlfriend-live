#!/usr/bin/env python3
"""AVTR-1 sidecar that publishes synchronized HTTP-FLV for the browser."""

from __future__ import annotations

import asyncio
import io
import json
import os
import random
import time
import wave
from collections import deque
from fractions import Fraction

import aiohttp
import av
import numpy as np
from aiohttp import web

RENDERER = os.environ.get("AVTR1_URL", "http://127.0.0.1:18012").rstrip("/")
HOST = os.environ.get("AVATAR_GW_HOST", "127.0.0.1")
PORT = int(os.environ.get("AVATAR_GW_PORT", "18011"))
AVATAR_ID = os.environ.get("AVTR1_AVATAR_ID", "xiaoya_locket")
BG_ID = os.environ.get("AVTR1_BG_ID", "plain_white")
H264_BITRATE = int(os.environ.get("AVTR1_H264_BITRATE", "1800000"))
SPEECH_START_BUFFER_MS = max(
    420, int(os.environ.get("AVATAR_TEE_PREROLL_MS", "800"))
)
SPEECH_REBUFFER_STEP_MS = max(
    100, int(os.environ.get("AVTR1_AUDIO_REBUFFER_STEP_MS", "200"))
)
SPEECH_MAX_BUFFER_MS = max(
    SPEECH_START_BUFFER_MS,
    int(os.environ.get("AVTR1_AUDIO_MAX_BUFFER_MS", "1400")),
)
AV_OUTPUT_RESERVOIR_MS = max(
    200, int(os.environ.get("AVTR1_OUTPUT_RESERVOIR_MS", "480"))
)
CFG_SELF_AUDIO = float(os.environ.get("AVTR1_CFG_SELF_AUDIO", "2.3"))
CFG_OTHER_AUDIO = float(os.environ.get("AVTR1_CFG_OTHER_AUDIO", "2.0"))
CFG_KP = float(os.environ.get("AVTR1_CFG_KP", "3.0"))
NOISE_ALPHA = float(os.environ.get("AVTR1_NOISE_ALPHA", "1.5"))
NOISE_TRUNC_Z = float(os.environ.get("AVTR1_NOISE_TRUNC_Z", "1.0"))
IDLE_NOISE_ALPHA = float(os.environ.get("AVTR1_IDLE_NOISE_ALPHA", "2.0"))
IDLE_NOISE_TRUNC_Z = float(os.environ.get("AVTR1_IDLE_NOISE_TRUNC_Z", "1.2"))
MOTION_AUDIO_RMS = max(1.0, float(os.environ.get("AVTR1_MOTION_AUDIO_RMS", "80")))
MOTION_LISTEN_RMS = max(
    MOTION_AUDIO_RMS, float(os.environ.get("AVTR1_MOTION_LISTEN_RMS", "450"))
)
MOTION_ACTIVE_HOLD_SECONDS = max(
    0.0, float(os.environ.get("AVTR1_MOTION_ACTIVE_HOLD_SECONDS", "1.0"))
)
BLINK_ENABLED = os.environ.get(
    "AVTR1_BLINK_ENABLED", os.environ.get("AVTR1_IDLE_BLINK_ENABLED", "1")
).lower() not in {"0", "false", "off", "no"}
BLINK_MIN_SECONDS = max(
    1.5,
    float(
        os.environ.get(
            "AVTR1_BLINK_MIN_SECONDS",
            os.environ.get("AVTR1_IDLE_BLINK_MIN_SECONDS", "2.4"),
        )
    ),
)
BLINK_MAX_SECONDS = max(
    BLINK_MIN_SECONDS,
    float(
        os.environ.get(
            "AVTR1_BLINK_MAX_SECONDS",
            os.environ.get("AVTR1_IDLE_BLINK_MAX_SECONDS", "6.8"),
        )
    ),
)
BLINK_STRENGTH = min(
    1.5,
    max(
        0.0,
        float(
            os.environ.get(
                "AVTR1_BLINK_STRENGTH",
                os.environ.get("AVTR1_IDLE_BLINK_STRENGTH", "1.08"),
            )
        ),
    ),
)
SPEECH_BLINK_STRENGTH = min(
    1.5,
    max(0.0, float(os.environ.get("AVTR1_BLINK_SPEECH_STRENGTH", str(BLINK_STRENGTH)))),
)
BLINK_SPEECH_INTERVAL_SCALE = min(
    1.2, max(0.35, float(os.environ.get("AVTR1_BLINK_SPEECH_INTERVAL_SCALE", "0.82")))
)
BLINK_DOUBLE_PROBABILITY = min(
    0.35, max(0.0, float(os.environ.get("AVTR1_BLINK_DOUBLE_PROBABILITY", "0.08")))
)
BLINK_PARTIAL_PROBABILITY = min(
    0.4, max(0.0, float(os.environ.get("AVTR1_BLINK_PARTIAL_PROBABILITY", "0.28")))
)
IDLE_BREATH_ENABLED = os.environ.get("AVTR1_IDLE_BREATH_ENABLED", "1").lower() not in {
    "0",
    "false",
    "off",
    "no",
}
IDLE_BREATH_POSE_DEGREES = min(
    0.8, max(0.0, float(os.environ.get("AVTR1_IDLE_BREATH_POSE_DEGREES", "0.65")))
)
IDLE_BREATH_PITCH_RATIO = min(
    1.0, max(-1.0, float(os.environ.get("AVTR1_IDLE_BREATH_PITCH_RATIO", "0.08")))
)
IDLE_BREATH_YAW_RATIO = min(
    1.0, max(-1.0, float(os.environ.get("AVTR1_IDLE_BREATH_YAW_RATIO", "1.0")))
)
IDLE_BREATH_ROLL_RATIO = min(
    1.0, max(-1.0, float(os.environ.get("AVTR1_IDLE_BREATH_ROLL_RATIO", "-0.12")))
)
IDLE_BREATH_PRIMARY_SECONDS = min(
    12.0, max(2.5, float(os.environ.get("AVTR1_IDLE_BREATH_PRIMARY_SECONDS", "4.0")))
)
IDLE_BREATH_DRIFT_SECONDS = min(
    16.0, max(4.0, float(os.environ.get("AVTR1_IDLE_BREATH_DRIFT_SECONDS", "9.1")))
)
IDLE_BREATH_DRIFT_MIX = min(
    0.5, max(0.0, float(os.environ.get("AVTR1_IDLE_BREATH_DRIFT_MIX", "0.30")))
)
IDLE_BREATH_FADE_IN_STEP = min(
    1.0, max(0.01, float(os.environ.get("AVTR1_IDLE_BREATH_FADE_IN_STEP", "0.08")))
)
IDLE_BREATH_FADE_OUT_STEP = min(
    1.0, max(0.01, float(os.environ.get("AVTR1_IDLE_BREATH_FADE_OUT_STEP", "0.18")))
)
MOTION_CONFIG_PATH = os.path.realpath(
    os.environ.get(
        "AVTR1_MOTION_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "..", "data", "avtr_motion.json"),
    )
)

# Controls actually consumed by the renderer. Bounds mirror its own clamps so
# the UI never offers values that AVTR-1 would silently discard.
MOTION_FIELDS = {
    "blink_enabled": ("bool", None, None),
    "idle_blink_min_seconds": ("float", 1.5, 12.0),
    "idle_blink_max_seconds": ("float", 1.5, 15.0),
    "idle_blink_strength": ("float", 0.0, 1.5),
    "idle_blink_double_probability": ("float", 0.0, 0.35),
    "idle_blink_partial_probability": ("float", 0.0, 0.4),
    "idle_motion_enabled": ("bool", None, None),
    "idle_head_amplitude_degrees": ("float", 0.0, 0.8),
    "idle_head_cycle_seconds": ("float", 2.5, 12.0),
    "idle_drift_cycle_seconds": ("float", 4.0, 16.0),
    "idle_drift_mix": ("float", 0.0, 0.5),
    "idle_pitch_ratio": ("float", -1.0, 1.0),
    "idle_yaw_ratio": ("float", -1.0, 1.0),
    "idle_roll_ratio": ("float", -1.0, 1.0),
    "idle_noise_alpha": ("float", 0.0, 4.0),
    "idle_noise_trunc_z": ("float", 0.0, 2.0),
    "speaking_blink_interval_scale": ("float", 0.35, 1.2),
    "speaking_blink_strength": ("float", 0.0, 1.5),
    "speaking_motion_strength": ("float", 0.0, 5.0),
    "listening_motion_strength": ("float", 0.0, 5.0),
    "speaking_noise_alpha": ("float", 0.0, 4.0),
    "speaking_noise_trunc_z": ("float", 0.0, 2.0),
    "speaking_motion_hold_seconds": ("float", 0.0, 3.0),
}


def _motion_config() -> dict[str, bool | float]:
    return {
        "blink_enabled": BLINK_ENABLED,
        "idle_blink_min_seconds": BLINK_MIN_SECONDS,
        "idle_blink_max_seconds": BLINK_MAX_SECONDS,
        "idle_blink_strength": BLINK_STRENGTH,
        "idle_blink_double_probability": BLINK_DOUBLE_PROBABILITY,
        "idle_blink_partial_probability": BLINK_PARTIAL_PROBABILITY,
        "idle_motion_enabled": IDLE_BREATH_ENABLED,
        "idle_head_amplitude_degrees": IDLE_BREATH_POSE_DEGREES,
        "idle_head_cycle_seconds": IDLE_BREATH_PRIMARY_SECONDS,
        "idle_drift_cycle_seconds": IDLE_BREATH_DRIFT_SECONDS,
        "idle_drift_mix": IDLE_BREATH_DRIFT_MIX,
        "idle_pitch_ratio": IDLE_BREATH_PITCH_RATIO,
        "idle_yaw_ratio": IDLE_BREATH_YAW_RATIO,
        "idle_roll_ratio": IDLE_BREATH_ROLL_RATIO,
        "idle_noise_alpha": IDLE_NOISE_ALPHA,
        "idle_noise_trunc_z": IDLE_NOISE_TRUNC_Z,
        "speaking_blink_interval_scale": BLINK_SPEECH_INTERVAL_SCALE,
        "speaking_blink_strength": SPEECH_BLINK_STRENGTH,
        "speaking_motion_strength": CFG_SELF_AUDIO,
        "listening_motion_strength": CFG_OTHER_AUDIO,
        "speaking_noise_alpha": NOISE_ALPHA,
        "speaking_noise_trunc_z": NOISE_TRUNC_Z,
        "speaking_motion_hold_seconds": MOTION_ACTIVE_HOLD_SECONDS,
    }


def _validated_motion_config(payload: object) -> dict[str, bool | float]:
    if not isinstance(payload, dict):
        raise ValueError("motion config must be an object")
    unknown = set(payload) - set(MOTION_FIELDS)
    if unknown:
        raise ValueError(f"unknown motion fields: {', '.join(sorted(unknown))}")
    values = _motion_config()
    for key, raw in payload.items():
        kind, minimum, maximum = MOTION_FIELDS[key]
        if kind == "bool":
            if not isinstance(raw, bool):
                raise ValueError(f"{key} must be boolean")
            values[key] = raw
            continue
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be numeric") from None
        if not np.isfinite(value) or value < minimum or value > maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        values[key] = value
    if values["idle_blink_max_seconds"] < values["idle_blink_min_seconds"]:
        raise ValueError("idle blink maximum must not be less than minimum")
    return values


def _apply_motion_config(values: dict[str, bool | float]) -> None:
    global BLINK_ENABLED, BLINK_MIN_SECONDS, BLINK_MAX_SECONDS, BLINK_STRENGTH
    global BLINK_DOUBLE_PROBABILITY, BLINK_PARTIAL_PROBABILITY
    global BLINK_SPEECH_INTERVAL_SCALE, SPEECH_BLINK_STRENGTH
    global IDLE_BREATH_ENABLED, IDLE_BREATH_POSE_DEGREES
    global IDLE_BREATH_PRIMARY_SECONDS, IDLE_BREATH_DRIFT_SECONDS
    global IDLE_BREATH_DRIFT_MIX, IDLE_BREATH_PITCH_RATIO
    global IDLE_BREATH_YAW_RATIO, IDLE_BREATH_ROLL_RATIO
    global IDLE_NOISE_ALPHA, IDLE_NOISE_TRUNC_Z
    global CFG_SELF_AUDIO, CFG_OTHER_AUDIO, NOISE_ALPHA, NOISE_TRUNC_Z
    global MOTION_ACTIVE_HOLD_SECONDS, next_blink_at
    BLINK_ENABLED = bool(values["blink_enabled"])
    BLINK_MIN_SECONDS = float(values["idle_blink_min_seconds"])
    BLINK_MAX_SECONDS = float(values["idle_blink_max_seconds"])
    BLINK_STRENGTH = float(values["idle_blink_strength"])
    BLINK_DOUBLE_PROBABILITY = float(values["idle_blink_double_probability"])
    BLINK_PARTIAL_PROBABILITY = float(values["idle_blink_partial_probability"])
    IDLE_BREATH_ENABLED = bool(values["idle_motion_enabled"])
    IDLE_BREATH_POSE_DEGREES = float(values["idle_head_amplitude_degrees"])
    IDLE_BREATH_PRIMARY_SECONDS = float(values["idle_head_cycle_seconds"])
    IDLE_BREATH_DRIFT_SECONDS = float(values["idle_drift_cycle_seconds"])
    IDLE_BREATH_DRIFT_MIX = float(values["idle_drift_mix"])
    IDLE_BREATH_PITCH_RATIO = float(values["idle_pitch_ratio"])
    IDLE_BREATH_YAW_RATIO = float(values["idle_yaw_ratio"])
    IDLE_BREATH_ROLL_RATIO = float(values["idle_roll_ratio"])
    IDLE_NOISE_ALPHA = float(values["idle_noise_alpha"])
    IDLE_NOISE_TRUNC_Z = float(values["idle_noise_trunc_z"])
    BLINK_SPEECH_INTERVAL_SCALE = float(values["speaking_blink_interval_scale"])
    SPEECH_BLINK_STRENGTH = float(values["speaking_blink_strength"])
    CFG_SELF_AUDIO = float(values["speaking_motion_strength"])
    CFG_OTHER_AUDIO = float(values["listening_motion_strength"])
    NOISE_ALPHA = float(values["speaking_noise_alpha"])
    NOISE_TRUNC_Z = float(values["speaking_noise_trunc_z"])
    MOTION_ACTIVE_HOLD_SECONDS = float(values["speaking_motion_hold_seconds"])
    blink_frames.clear()
    next_blink_at = time.monotonic() + random.uniform(BLINK_MIN_SECONDS, BLINK_MAX_SECONDS)


def _save_motion_config(values: dict[str, bool | float]) -> None:
    directory = os.path.dirname(MOTION_CONFIG_PATH)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    temporary = f"{MOTION_CONFIG_PATH}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, MOTION_CONFIG_PATH)


def _load_saved_motion_config() -> None:
    if not os.path.isfile(MOTION_CONFIG_PATH):
        return
    try:
        with open(MOTION_CONFIG_PATH, encoding="utf-8") as handle:
            _apply_motion_config(_validated_motion_config(json.load(handle)))
    except Exception as exc:
        print(f"[avtr1-gw] ignoring invalid motion config: {exc}", flush=True)
BACKGROUND_MUSIC_ENABLED = os.environ.get(
    "BACKGROUND_MUSIC_ENABLED", "1"
).lower() not in {"0", "false", "off", "no"}
BACKGROUND_MUSIC_DIR = os.path.realpath(os.environ.get("BACKGROUND_MUSIC_DIR", "."))
BACKGROUND_MUSIC_VOLUME = min(
    1.0, max(0.0, float(os.environ.get("BACKGROUND_MUSIC_VOLUME", "0.16")))
)
BACKGROUND_MUSIC_DUCK_VOLUME = min(
    BACKGROUND_MUSIC_VOLUME,
    max(0.0, float(os.environ.get("BACKGROUND_MUSIC_DUCK_VOLUME", "0.04"))),
)
BACKGROUND_MUSIC_USER_RMS = max(
    1.0, float(os.environ.get("BACKGROUND_MUSIC_USER_RMS", "450"))
)

SAMPLE_RATE = 16_000
CHUNK_SIZE = 5
FRAME_LEN = 640
AUDIO_SHIFT = 80
CURRENT_SAMPLES = CHUNK_SIZE * FRAME_LEN
FUTURE_SAMPLES = CHUNK_SIZE * FRAME_LEN + AUDIO_SHIFT
WINDOW_SAMPLES = CURRENT_SAMPLES + FUTURE_SAMPLES
PCM_PACKET_BYTES = 640
MAX_SPEECH_SECONDS = max(30, int(os.environ.get("AVTR1_MAX_SPEECH_SECONDS", "90")))
MAX_SPEECH_BYTES = SAMPLE_RATE * 2 * MAX_SPEECH_SECONDS
SPEECH_START_BUFFER_BYTES = SAMPLE_RATE * 2 * SPEECH_START_BUFFER_MS // 1000
SPEECH_MAX_BUFFER_BYTES = SAMPLE_RATE * 2 * SPEECH_MAX_BUFFER_MS // 1000
SPEECH_REBUFFER_STEP_BYTES = SAMPLE_RATE * 2 * SPEECH_REBUFFER_STEP_MS // 1000
OUTPUT_AV_TARGET_FRAMES = max(CHUNK_SIZE, AV_OUTPUT_RESERVOIR_MS // 40)

last_frame_at = 0.0
last_speech_input_at = 0.0
last_user_voice_at = 0.0
last_motion_audio_at = 0.0
last_blink_at = 0.0
next_blink_at = time.monotonic() + random.uniform(BLINK_MIN_SECONDS, BLINK_MAX_SECONDS)
blink_frames: deque[float] = deque()
breath_mix = 0.0
connected = False
state_blob: bytes | None = None
state_avatar_id: str | None = None
speech_pcm = bytearray()
speech_finished = False
speech_turn_active = False
speech_playing = False
speech_rebuffering = False
speech_output_ready = False
speech_dynamic_buffer_bytes = SPEECH_START_BUFFER_BYTES
speech_turn_underruns = 0
speech_buffer_underruns = 0
speech_silence_inserted_ms = 0
speech_turns_completed = 0
speech_stable_turns = 0
last_tts_metrics: dict[str, float | int] = {}
listen_pcm = bytearray()
buf_lock = asyncio.Lock()
# Queue -> whether this viewer requested the mixed background-music variant.
flv_subscribers: dict[asyncio.Queue, bool] = {}
# A single frame queue keeps the rendered face and its exact 40ms audio slice
# inseparable. Separate video/audio queues can drift when either side drops.
av_pace_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
h264_encoder: H264Encoder | None = None
h264_bytes = 0
video_epoch = 0
video_frames_rendered = 0
video_frames_published = 0
video_frames_held = 0
video_queue_drops = 0
publisher_late_ticks = 0
audio_output_underruns = 0
render_batches = 0
render_deadline_misses = 0
render_errors = 0
render_short_batches = 0
render_durations_ms: deque[float] = deque(maxlen=900)
renderer_session: aiohttp.ClientSession | None = None
flv_muxer_music: FlvMuxer | None = None
flv_muxer_voice: FlvMuxer | None = None


class BackgroundMusic:
    """Decode a local MP3 playlist once, then mix it into the shared FLV audio.

    Music is added after AVTR-1 has generated the face motion, so instrumental
    energy can never drive the mouth. Gain changes use a short attack and a
    slower release to avoid audible pumping around pauses between words.
    """

    def __init__(self) -> None:
        self.tracks: list[tuple[str, np.ndarray]] = []
        self.track_index = 0
        self.sample_index = 0
        self.gain = BACKGROUND_MUSIC_VOLUME

    @property
    def available(self) -> bool:
        return bool(self.tracks)

    @property
    def track_name(self) -> str | None:
        return self.tracks[self.track_index][0] if self.tracks else None

    def load(self) -> None:
        self.tracks.clear()
        self.track_index = 0
        self.sample_index = 0
        self.gain = BACKGROUND_MUSIC_VOLUME
        if not BACKGROUND_MUSIC_ENABLED or not os.path.isdir(BACKGROUND_MUSIC_DIR):
            return
        names = sorted(
            (
                name
                for name in os.listdir(BACKGROUND_MUSIC_DIR)
                if name.lower().endswith(".mp3")
                and os.path.isfile(os.path.join(BACKGROUND_MUSIC_DIR, name))
            ),
            key=str.casefold,
        )
        for name in names:
            path = os.path.join(BACKGROUND_MUSIC_DIR, name)
            chunks: list[np.ndarray] = []
            try:
                with av.open(path) as container:
                    stream = container.streams.audio[0]
                    resampler = av.AudioResampler(
                        format="s16", layout="mono", rate=SAMPLE_RATE
                    )
                    for frame in container.decode(stream):
                        for output in resampler.resample(frame):
                            chunks.append(output.to_ndarray().reshape(-1).copy())
                    for output in resampler.resample(None):
                        chunks.append(output.to_ndarray().reshape(-1).copy())
                samples = (
                    np.concatenate(chunks).astype(np.int16, copy=False)
                    if chunks
                    else None
                )
                if samples is not None and samples.size:
                    self.tracks.append((name, samples))
            except Exception as exc:
                print(f"[avtr1-gw] skip background music {name}: {exc}", flush=True)
        if self.tracks:
            duration = sum(samples.size for _, samples in self.tracks) / SAMPLE_RATE
            print(
                f"[avtr1-gw] background music tracks={len(self.tracks)} "
                f"duration={duration:.1f}s volume={BACKGROUND_MUSIC_VOLUME:.2f} "
                f"duck={BACKGROUND_MUSIC_DUCK_VOLUME:.2f}",
                flush=True,
            )

    def _take(self, count: int) -> np.ndarray:
        output = np.empty(count, dtype=np.int16)
        written = 0
        while written < count and self.tracks:
            _, samples = self.tracks[self.track_index]
            available = samples.size - self.sample_index
            size = min(count - written, available)
            output[written : written + size] = samples[
                self.sample_index : self.sample_index + size
            ]
            written += size
            self.sample_index += size
            if self.sample_index >= samples.size:
                self.track_index = (self.track_index + 1) % len(self.tracks)
                self.sample_index = 0
        if written < count:
            output[written:] = 0
        return output

    def mix(self, voice_pcm: bytes, *, ducked: bool) -> bytes:
        if not self.tracks or not voice_pcm:
            return voice_pcm
        voice = np.frombuffer(voice_pcm, dtype=np.int16)
        music = self._take(voice.size).astype(np.float32)
        target = BACKGROUND_MUSIC_DUCK_VOLUME if ducked else BACKGROUND_MUSIC_VOLUME
        duration = max(voice.size / SAMPLE_RATE, 0.001)
        tau = 0.12 if target < self.gain else 0.75
        end_gain = target + (self.gain - target) * np.exp(-duration / tau)
        gains = np.linspace(self.gain, end_gain, num=voice.size, dtype=np.float32)
        self.gain = float(end_gain)
        mixed = voice.astype(np.float32) + music * gains
        return np.clip(mixed, -32768, 32767).astype(np.int16).tobytes()


background_music = BackgroundMusic()


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
        return [
            (bytes(packet), bool(packet.is_keyframe))
            for packet in self.ctx.encode(frame)
        ]


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
            header = self._tag(
                9, bytes((0x17, 0x00, 0x00, 0x00, 0x00)) + record, self.timestamp_ms
            )
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
                tags.append(
                    self._tag(8, bytes((0xAE, 0x01)) + bytes(packet), self.timestamp_ms)
                )
        return tags

    def advance(self, ms: int = 40) -> None:
        self.timestamp_ms += ms


def publish_flv(data: bytes, *, music: bool) -> None:
    if not data:
        return
    for q, wants_music in tuple(flv_subscribers.items()):
        if wants_music != music:
            continue
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


def enqueue_av_frame(data) -> None:
    global video_queue_drops
    if av_pace_queue.full():
        try:
            av_pace_queue.get_nowait()
            video_queue_drops += 1
        except asyncio.QueueEmpty:
            pass
    try:
        av_pace_queue.put_nowait(data)
    except asyncio.QueueFull:
        video_queue_drops += 1


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
    global h264_encoder, h264_bytes
    global video_frames_published, video_frames_held, publisher_late_ticks
    global audio_output_underruns, speech_output_ready
    loop = asyncio.get_running_loop()
    deadline = loop.time()
    last_video: tuple[int, bytes, int, int] | None = None
    active_epoch = -1
    while True:
        now = loop.time()
        if now - deadline > 0.06:
            publisher_late_ticks += 1
            # Never burst several overdue frames: restart the wall clock while
            # preserving FLV/RTP timestamps at an exact 40ms cadence.
            deadline = now
        # Rendering owns 200ms batches while publishing owns a strict 40ms
        # clock. Hold the previous silent/idle frame until enough synchronized
        # speech frames are ready; then transient >200ms inference spikes are
        # absorbed without putting silence between spoken words.
        if speech_playing and not speech_output_ready:
            if av_pace_queue.qsize() >= OUTPUT_AV_TARGET_FRAMES:
                speech_output_ready = True
            else:
                fresh = None
        if not speech_playing or speech_output_ready:
            try:
                fresh = av_pace_queue.get_nowait()
            except asyncio.QueueEmpty:
                fresh = None

        audio_chunks = (bytes(PCM_PACKET_BYTES), bytes(PCM_PACKET_BYTES))
        if fresh is not None:
            epoch, raw, width, height, audio_chunks = fresh
            if epoch != active_epoch:
                active_epoch = epoch
                h264_encoder = None
                last_video = None
            last_video = (epoch, raw, width, height)
        elif last_video is not None:
            video_frames_held += 1
            if speech_playing and speech_output_ready:
                audio_output_underruns += 1

        if last_video is not None:
            _epoch, raw, width, height = last_video
            if h264_encoder is None or (
                h264_encoder.width,
                h264_encoder.height,
            ) != (width, height):
                h264_encoder = H264Encoder(width, height)
                print(
                    f"[avtr1-gw] H.264 {width}x{height} 25fps bitrate={H264_BITRATE}",
                    flush=True,
                )
            packets = h264_encoder.encode(raw)
            h264_bytes += sum(len(packet) for packet, _ in packets)
            video_frames_published += 1
            if flv_muxer_music is not None and flv_muxer_voice is not None:
                for packet, keyframe in packets:
                    for tag in flv_muxer_music.video_tags(packet, keyframe):
                        publish_flv(tag, music=True)
                    for tag in flv_muxer_voice.video_tags(packet, keyframe):
                        publish_flv(tag, music=False)
        for last_audio in audio_chunks:
            now = time.monotonic()
            ducked = (
                bool(speech_pcm)
                or now - last_speech_input_at < 1.6
                or now - last_user_voice_at < 0.8
            )
            output_audio = background_music.mix(last_audio, ducked=ducked)
            if flv_muxer_music is not None and flv_muxer_voice is not None:
                for tag in flv_muxer_music.audio_tags(output_audio):
                    publish_flv(tag, music=True)
                for tag in flv_muxer_voice.audio_tags(last_audio):
                    publish_flv(tag, music=False)
        if flv_muxer_music is not None and flv_muxer_voice is not None:
            flv_muxer_music.advance(40)
            flv_muxer_voice.advance(40)
        deadline += 0.04
        await asyncio.sleep(max(0.0, deadline - loop.time()))


def _window_from_speech(buf: bytearray) -> tuple[bytes, bytes, bytes, bool]:
    """Return one continuous 200ms speech slice plus AVTR look-ahead.

    Playback starts only after the whole-turn reservoir reaches its dynamic
    watermark. If synthesis later falls behind, enter one rebuffering period
    and wait for the reservoir to recover instead of alternating 200ms speech
    and 200ms silence forever.
    """
    global speech_finished, speech_turn_active, speech_playing, speech_rebuffering
    global speech_dynamic_buffer_bytes, speech_turn_underruns
    global speech_buffer_underruns, speech_silence_inserted_ms
    global speech_turns_completed, speech_stable_turns

    def silence() -> tuple[bytes, bytes, bytes, bool]:
        current = bytes(CURRENT_SAMPLES * 2)
        return current, bytes(FUTURE_SAMPLES * 2), current, False

    def finish_turn() -> None:
        nonlocal buf
        global speech_finished, speech_turn_active, speech_playing, speech_rebuffering
        global speech_dynamic_buffer_bytes, speech_turn_underruns
        global speech_turns_completed, speech_stable_turns
        speech_finished = False
        speech_turn_active = False
        speech_playing = False
        speech_rebuffering = False
        speech_turns_completed += 1
        if speech_turn_underruns == 0:
            speech_stable_turns += 1
            if speech_stable_turns >= 3:
                speech_dynamic_buffer_bytes = max(
                    SPEECH_START_BUFFER_BYTES,
                    speech_dynamic_buffer_bytes - SPEECH_REBUFFER_STEP_BYTES,
                )
                speech_stable_turns = 0
        else:
            speech_stable_turns = 0
        speech_turn_underruns = 0

    need = WINDOW_SAMPLES * 2
    if not speech_turn_active and not buf:
        return silence()

    if not speech_playing:
        if not buf and speech_finished:
            finish_turn()
            return silence()
        if not speech_finished and len(buf) < speech_dynamic_buffer_bytes:
            if speech_rebuffering:
                speech_silence_inserted_ms += 200
            return silence()
        speech_playing = True
        speech_rebuffering = False

    if len(buf) >= need:
        window = bytes(buf[:need])
        del buf[: CURRENT_SAMPLES * 2]
    elif not speech_finished:
        speech_playing = False
        speech_rebuffering = True
        speech_turn_underruns += 1
        speech_buffer_underruns += 1
        speech_silence_inserted_ms += 200
        speech_dynamic_buffer_bytes = min(
            SPEECH_MAX_BUFFER_BYTES,
            speech_dynamic_buffer_bytes + SPEECH_REBUFFER_STEP_BYTES,
        )
        return silence()
    elif buf:
        window = bytes(buf[:need]) + bytes(max(0, need - len(buf)))
        consumed = min(len(buf), CURRENT_SAMPLES * 2)
        del buf[:consumed]
        if not buf:
            finish_turn()
    else:
        finish_turn()
        return silence()
    cur = window[: CURRENT_SAMPLES * 2]
    fut = window[CURRENT_SAMPLES * 2 :]
    return cur, fut, cur, True


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


def _pcm_rms(*chunks: bytes) -> float:
    """Return int16 PCM RMS without a Python-level sample loop."""
    usable = [
        chunk[: len(chunk) - len(chunk) % 2] for chunk in chunks if len(chunk) >= 2
    ]
    if not usable:
        return 0.0
    samples = np.frombuffer(b"".join(usable), dtype="<i2").astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _single_blink_profile(*, partial: bool = False) -> list[float]:
    """Return an asymmetric 25-fps blink: fast close, slower soft reopen."""
    close_frames = random.choice((2, 2, 2, 3))
    hold_frames = random.choice((0, 0, 0, 1))
    open_frames = random.choice((3, 3, 4, 4, 5))
    peak = random.uniform(0.48, 0.72) if partial else random.uniform(0.88, 1.02)
    closing = [peak * _smoothstep(i / close_frames) for i in range(1, close_frames + 1)]
    opening = [
        peak * (1.0 - _smoothstep(i / open_frames)) for i in range(1, open_frames + 1)
    ]
    return closing + [peak] * hold_frames + opening


def _new_blink_profile() -> list[float]:
    partial = random.random() < BLINK_PARTIAL_PROBABILITY
    profile = _single_blink_profile(partial=partial)
    if not partial and random.random() < BLINK_DOUBLE_PROBABILITY:
        profile.extend([0.0] * random.choice((2, 3, 4)))
        profile.extend(_single_blink_profile(partial=random.random() < 0.35))
    return profile


def _next_blink_delay(*, speaking: bool) -> float:
    # A beta distribution avoids metronomic uniform intervals while retaining
    # hard safety bounds. Conversation uses a shorter interval than quiet rest.
    delay = BLINK_MIN_SECONDS + (
        BLINK_MAX_SECONDS - BLINK_MIN_SECONDS
    ) * random.betavariate(1.7, 2.2)
    return delay * (BLINK_SPEECH_INTERVAL_SCALE if speaking else 1.0)


def _breath_weights(now: float, *, enabled: bool) -> list[float]:
    """Five continuous low-frequency samples; two periods prevent a loop feel."""
    if not enabled or IDLE_BREATH_POSE_DEGREES <= 0.0:
        return [0.0] * CHUNK_SIZE
    # Keep float64 here: monotonic clocks can already be millions of seconds
    # since boot, where float32 would quantize away the 40ms frame steps.
    times = now + np.arange(CHUNK_SIZE, dtype=np.float64) / 25.0
    slow = np.sin((2.0 * np.pi / IDLE_BREATH_PRIMARY_SECONDS) * times)
    drift = np.sin((2.0 * np.pi / IDLE_BREATH_DRIFT_SECONDS) * times + 1.1)
    slow_mix = 1.0 - IDLE_BREATH_DRIFT_MIX
    return ((slow * slow_mix + drift * IDLE_BREATH_DRIFT_MIX) * breath_mix).tolist()


async def render_loop() -> None:
    global \
        last_frame_at, \
        connected, \
        state_blob, \
        state_avatar_id, \
        renderer_session
    global last_motion_audio_at
    global last_blink_at, next_blink_at, breath_mix
    global render_batches, render_deadline_misses, render_errors, render_short_batches
    global video_frames_rendered
    loop = asyncio.get_running_loop()
    while True:
        t0 = loop.time()
        try:
            async with buf_lock:
                cur, fut, played, rendered_speech = _window_from_speech(speech_pcm)
                listen_cur, listen_fut = _window_from_listen(listen_pcm)
                avatar_id = AVATAR_ID
                epoch = video_epoch
                blob = state_blob if state_avatar_id == avatar_id else None
            now = time.monotonic()
            speech_active = _pcm_rms(cur, fut) >= MOTION_AUDIO_RMS
            listen_active = _pcm_rms(listen_cur, listen_fut) >= MOTION_LISTEN_RMS
            if speech_active or listen_active:
                last_motion_audio_at = now
            motion_active = now - last_motion_audio_at <= MOTION_ACTIVE_HOLD_SECONDS
            noise_alpha = NOISE_ALPHA if motion_active else IDLE_NOISE_ALPHA
            noise_trunc_z = NOISE_TRUNC_Z if motion_active else IDLE_NOISE_TRUNC_Z
            # Ease breathing out during speech and back in during quiet instead
            # of snapping the head pose at an audio boundary.
            breath_mix = (
                max(0.0, breath_mix - IDLE_BREATH_FADE_OUT_STEP)
                if motion_active
                else min(1.0, breath_mix + IDLE_BREATH_FADE_IN_STEP)
            )
            micro_pose_weights = _breath_weights(now, enabled=IDLE_BREATH_ENABLED)

            if BLINK_ENABLED and not blink_frames and now >= next_blink_at:
                blink_frames.extend(_new_blink_profile())
                last_blink_at = now
                next_blink_at = (
                    now
                    + len(blink_frames) / 25.0
                    + _next_blink_delay(speaking=motion_active)
                )
            blink_weights = [
                blink_frames.popleft() if blink_frames else 0.0
                for _ in range(CHUNK_SIZE)
            ]
            blink_strength = (
                SPEECH_BLINK_STRENGTH if motion_active else BLINK_STRENGTH
            ) if any(blink_weights) else 0.0
            form = aiohttp.FormData()
            form.add_field(
                "current_chunk",
                cur,
                filename="cur.raw",
                content_type="application/octet-stream",
            )
            form.add_field(
                "future_chunk",
                fut,
                filename="fut.raw",
                content_type="application/octet-stream",
            )
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
                form.add_field(
                    "state",
                    blob,
                    filename="state.bin",
                    content_type="application/octet-stream",
                )
            params = {
                "avatar_id": avatar_id,
                "bg_id": BG_ID,
                "pixel_format": "yuv_i420",
                "cfg_self_audio": str(CFG_SELF_AUDIO),
                "cfg_other_audio": str(CFG_OTHER_AUDIO),
                "cfg_kp": str(CFG_KP),
                "noise_alpha": str(noise_alpha),
                "noise_trunc_z": str(noise_trunc_z),
                "blink_strength": str(blink_strength),
                "blink_weights": ",".join(f"{value:.4f}" for value in blink_weights),
                "micro_pose_degrees": str(IDLE_BREATH_POSE_DEGREES),
                "micro_pose_weights": ",".join(
                    f"{value:.4f}" for value in micro_pose_weights
                ),
                "micro_pose_pitch_ratio": str(IDLE_BREATH_PITCH_RATIO),
                "micro_pose_yaw_ratio": str(IDLE_BREATH_YAW_RATIO),
                "micro_pose_roll_ratio": str(IDLE_BREATH_ROLL_RATIO),
            }
            if renderer_session is None:
                raise RuntimeError("renderer HTTP session is not initialized")
            async with renderer_session.post(
                f"{RENDERER}/process-audio-v3", data=form, params=params
            ) as r:
                if r.status != 200:
                    body = await r.text()
                    print(f"[avtr1-gw] renderer {r.status}: {body[:300]}", flush=True)
                    connected = False
                    render_errors += 1
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
            elapsed = loop.time() - t0
            render_batches += 1
            render_durations_ms.append(elapsed * 1000.0)
            if elapsed > 0.2:
                render_deadline_misses += 1
            if n_frames < CHUNK_SIZE:
                render_short_batches += 1
            next_state = body[:state_len]
            frames = body[state_len:]
            async with buf_lock:
                if AVATAR_ID == avatar_id:
                    state_blob = next_state
                    state_avatar_id = avatar_id
                else:
                    # The administrator switched profiles while this inference
                    # was running. Never publish stale frames from the old face.
                    continue
            for i in range(n_frames):
                raw = frames[i * frame_len : (i + 1) * frame_len]
                if len(raw) != frame_len:
                    break
                last_frame_at = time.time()
                connected = True
                video_frames_rendered += 1
                off = i * 2 * PCM_PACKET_BYTES
                first_audio = (
                    played[off : off + PCM_PACKET_BYTES]
                    if rendered_speech
                    else bytes(PCM_PACKET_BYTES)
                )
                second_audio = (
                    played[off + PCM_PACKET_BYTES : off + 2 * PCM_PACKET_BYTES]
                    if rendered_speech
                    else bytes(PCM_PACKET_BYTES)
                )
                enqueue_av_frame(
                    (
                        epoch,
                        raw,
                        w,
                        h,
                        (
                            first_audio or bytes(PCM_PACKET_BYTES),
                            second_audio or bytes(PCM_PACKET_BYTES),
                        ),
                    )
                )
            # While a response is playing, use spare renderer capacity to keep
            # a bounded synchronized A/V reservoir. TTS can synthesize the next
            # sentence while the current sentence drains from this queue.
            fill_reservoir = rendered_speech and av_pace_queue.qsize() < OUTPUT_AV_TARGET_FRAMES
            await asyncio.sleep(0 if fill_reservoir else max(0.0, 0.2 - elapsed))
        except Exception as exc:
            connected = False
            render_errors += 1
            async with buf_lock:
                state_blob = None
                state_avatar_id = None
            print("[avtr1-gw] render failed:", exc, flush=True)
            await asyncio.sleep(0.5)


async def append_speech(pcm: bytes) -> None:
    global last_speech_input_at, speech_finished
    global speech_turn_active, speech_playing, speech_rebuffering, speech_turn_underruns
    global speech_output_ready
    if not pcm:
        return
    last_speech_input_at = time.monotonic()
    async with buf_lock:
        if not speech_turn_active:
            speech_turn_active = True
            speech_playing = False
            speech_rebuffering = False
            speech_turn_underruns = 0
            speech_output_ready = False
            while not av_pace_queue.empty():
                try:
                    av_pace_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        speech_finished = False
        speech_pcm.extend(pcm)
        overflow = len(speech_pcm) - MAX_SPEECH_BYTES
        if overflow > 0:
            del speech_pcm[: overflow - (overflow % 2)]


async def append_listen(pcm: bytes) -> None:
    global last_user_voice_at
    if not pcm:
        return
    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=np.int16)
    if samples.size:
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        if rms >= BACKGROUND_MUSIC_USER_RMS:
            last_user_voice_at = time.monotonic()
    async with buf_lock:
        listen_pcm.extend(pcm)
        overflow = len(listen_pcm) - MAX_SPEECH_BYTES
        if overflow > 0:
            del listen_pcm[: overflow - (overflow % 2)]


AVATAR_LABELS = {
    "xiaoya": "小雅",
    "xiaoya_idle": "暖光正脸",
    "xiaoya_beach_close": "海边近景",
    "xiaoya_beach": "海边",
    "xiaoya_locket": "白背心",
    "sauna_portrait": "桑拿正脸",
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
    preferred = [
        "xiaoya_locket",
        "xiaoya",
        "xiaoya_idle",
        "xiaoya_beach_close",
        "xiaoya_beach",
        "sauna_portrait",
    ]
    ordered = [item for item in preferred if item in ids]
    return ordered or [AVATAR_ID], loaded


async def handle_status(_request):
    def percentile(values: deque[float], value: float) -> float | None:
        if not values:
            return None
        return round(float(np.percentile(np.asarray(values), value)), 2)

    speaking = speech_turn_active or time.monotonic() - last_speech_input_at < 1.6
    music_ducked = speaking or time.monotonic() - last_user_voice_at < 0.8
    return web.json_response(
        {
            "connected": connected and last_frame_at > 0,
            "backend": "avtr1",
            "avatar_id": AVATAR_ID,
            "age_ms": int((time.time() - last_frame_at) * 1000)
            if last_frame_at
            else None,
            "speech_ms": int(len(speech_pcm) / 2 / SAMPLE_RATE * 1000),
            "speaking": speaking,
            "audio_buffer": {
                "state": (
                    "playing"
                    if speech_playing
                    else "rebuffering"
                    if speech_rebuffering
                    else "buffering"
                    if speech_turn_active
                    else "idle"
                ),
                "queued_ms": int(len(speech_pcm) / 2 / SAMPLE_RATE * 1000),
                "output_reservoir_ms": av_pace_queue.qsize() * 40,
                "output_buffering": speech_playing and not speech_output_ready,
                "watermark_ms": int(
                    speech_dynamic_buffer_bytes / 2 / SAMPLE_RATE * 1000
                ),
                "start_watermark_ms": SPEECH_START_BUFFER_MS,
                "max_watermark_ms": SPEECH_MAX_BUFFER_MS,
                "underruns": speech_buffer_underruns,
                "output_underruns": audio_output_underruns,
                "inserted_silence_ms": speech_silence_inserted_ms,
                "turns_completed": speech_turns_completed,
                "last_tts": last_tts_metrics,
            },
            "render": {
                "batches": render_batches,
                "duration_p50_ms": percentile(render_durations_ms, 50),
                "duration_p95_ms": percentile(render_durations_ms, 95),
                "duration_p99_ms": percentile(render_durations_ms, 99),
                "deadline_misses": render_deadline_misses,
                "errors": render_errors,
                "short_batches": render_short_batches,
                "frames_rendered": video_frames_rendered,
                "frames_published": video_frames_published,
                "held_frames": video_frames_held,
                "queue_drops": video_queue_drops,
                "publisher_late_ticks": publisher_late_ticks,
                "queue_frames": av_pace_queue.qsize(),
            },
            "background_music": {
                "enabled": background_music.available,
                "track": background_music.track_name,
                "volume": BACKGROUND_MUSIC_VOLUME,
                "duck_volume": BACKGROUND_MUSIC_DUCK_VOLUME,
                "ducked": music_ducked,
            },
            "listen_ms": int(len(listen_pcm) / 2 / SAMPLE_RATE * 1000),
            "flv_clients": len(flv_subscribers),
            "h264_bitrate": H264_BITRATE,
            "h264_bytes": h264_bytes,
            "motion": {
                "active_noise_alpha": NOISE_ALPHA,
                "active_noise_trunc_z": NOISE_TRUNC_Z,
                "idle_noise_alpha": IDLE_NOISE_ALPHA,
                "idle_noise_trunc_z": IDLE_NOISE_TRUNC_Z,
                "audio_rms_threshold": MOTION_AUDIO_RMS,
                "listen_rms_threshold": MOTION_LISTEN_RMS,
                "active_hold_seconds": MOTION_ACTIVE_HOLD_SECONDS,
                "blink_enabled": BLINK_ENABLED,
                "blink_strength": BLINK_STRENGTH,
                "blink_speech_strength": SPEECH_BLINK_STRENGTH,
                "blink_min_seconds": BLINK_MIN_SECONDS,
                "blink_max_seconds": BLINK_MAX_SECONDS,
                "blink_speech_interval_scale": BLINK_SPEECH_INTERVAL_SCALE,
                "blink_double_probability": BLINK_DOUBLE_PROBABILITY,
                "blink_partial_probability": BLINK_PARTIAL_PROBABILITY,
                "blink_frames_remaining": len(blink_frames),
                "last_blink_age_ms": (
                    int((time.monotonic() - last_blink_at) * 1000)
                    if last_blink_at
                    else None
                ),
                "next_blink_ms": max(0, int((next_blink_at - time.monotonic()) * 1000)),
                "idle_breath_enabled": IDLE_BREATH_ENABLED,
                "idle_breath_pose_degrees": IDLE_BREATH_POSE_DEGREES,
                "idle_breath_pitch_ratio": IDLE_BREATH_PITCH_RATIO,
                "idle_breath_yaw_ratio": IDLE_BREATH_YAW_RATIO,
                "idle_breath_roll_ratio": IDLE_BREATH_ROLL_RATIO,
                "idle_breath_primary_seconds": IDLE_BREATH_PRIMARY_SECONDS,
                "idle_breath_drift_seconds": IDLE_BREATH_DRIFT_SECONDS,
                "idle_breath_drift_mix": IDLE_BREATH_DRIFT_MIX,
                "idle_breath_fade_in_step": IDLE_BREATH_FADE_IN_STEP,
                "idle_breath_fade_out_step": IDLE_BREATH_FADE_OUT_STEP,
                "idle_breath_mix": round(breath_mix, 3),
            },
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
    global AVATAR_ID, state_blob, state_avatar_id, h264_encoder, next_blink_at
    global speech_finished, speech_turn_active, speech_playing, speech_rebuffering
    global speech_turn_underruns, speech_dynamic_buffer_bytes, speech_output_ready, video_epoch
    body = await request.json()
    avatar_id = str(body.get("avatar_id") or "").strip()
    if not avatar_id or "/" in avatar_id or ".." in avatar_id:
        return web.json_response(
            {"ok": False, "error": "invalid avatar_id"}, status=400
        )
    ids, _loaded = await _list_avatar_ids()
    if avatar_id not in ids:
        return web.json_response({"ok": False, "error": "unknown avatar"}, status=404)
    async with buf_lock:
        AVATAR_ID = avatar_id
        speech_pcm.clear()
        speech_finished = False
        speech_turn_active = False
        speech_playing = False
        speech_rebuffering = False
        speech_turn_underruns = 0
        speech_output_ready = False
        speech_dynamic_buffer_bytes = SPEECH_START_BUFFER_BYTES
        listen_pcm.clear()
        state_blob = None
        state_avatar_id = None
        h264_encoder = None
        video_epoch += 1
        # Show the newly selected portrait is alive soon after switching,
        # without blinking immediately on the first generated frame.
        blink_frames.clear()
        next_blink_at = time.monotonic() + random.uniform(1.2, 2.4)
    while not av_pace_queue.empty():
        try:
            av_pace_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    print(f"[avtr1-gw] avatar -> {AVATAR_ID}", flush=True)
    return web.json_response(
        {"ok": True, "avatar_id": AVATAR_ID, "label": _avatar_label(AVATAR_ID)}
    )


async def handle_motion_config(_request):
    return web.json_response({"ok": True, "motion": _motion_config()})


async def handle_set_motion_config(request):
    try:
        values = _validated_motion_config(await request.json())
        _apply_motion_config(values)
        _save_motion_config(values)
    except (ValueError, json.JSONDecodeError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    print("[avtr1-gw] motion config updated", flush=True)
    return web.json_response({"ok": True, "motion": values})


async def handle_livestream(request):
    wants_music = request.query.get("music", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
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
    selected_muxer = flv_muxer_music if wants_music else flv_muxer_voice
    if selected_muxer is not None:
        bootstrap = selected_muxer.bootstrap()
        if bootstrap:
            try:
                await resp.write(bootstrap)
                await resp.drain()
            except (ConnectionResetError, ConnectionError, asyncio.CancelledError):
                return resp
    flv_subscribers[q] = wants_music
    try:
        while True:
            chunk = await q.get()
            await resp.write(chunk)
            await resp.drain()
    except (
        ConnectionResetError,
        ConnectionError,
        asyncio.CancelledError,
        RuntimeError,
    ):
        pass
    finally:
        flv_subscribers.pop(q, None)
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


async def handle_audio_finish(_request):
    global speech_finished, last_tts_metrics
    async with buf_lock:
        speech_finished = True
        metrics: dict[str, float | int] = {}
        for key in ("audio_ms", "generation_ms", "rtf", "chunks", "max_gap_ms"):
            raw = _request.query.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if np.isfinite(value) and 0 <= value <= 3_600_000:
                metrics[key] = round(value, 3)
        # Direct WAV tests do not carry synthesis metrics; clear the previous
        # turn rather than presenting stale TTS performance as current data.
        last_tts_metrics = metrics
    return web.json_response({"ok": True, "remaining_bytes": len(speech_pcm)})


async def handle_listen_chunk(request):
    raw = await request.read()
    await append_listen(raw)
    return web.json_response({"ok": True, "bytes": len(raw)})


async def handle_listen_reset(_request):
    global last_user_voice_at
    async with buf_lock:
        listen_pcm.clear()
        last_user_voice_at = 0.0
    return web.json_response({"ok": True})


async def handle_interrupt(_request):
    global state_blob, last_speech_input_at, speech_finished
    global speech_turn_active, speech_playing, speech_rebuffering, speech_turn_underruns
    global speech_output_ready
    async with buf_lock:
        speech_pcm.clear()
        speech_finished = False
        speech_turn_active = False
        speech_playing = False
        speech_rebuffering = False
        speech_turn_underruns = 0
        speech_output_ready = False
        state_blob = None
        last_speech_input_at = 0.0
    while not av_pace_queue.empty():
        try:
            av_pace_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    return web.json_response({"ok": True})


async def on_startup(app):
    global renderer_session, flv_muxer_music, flv_muxer_voice
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=5, sock_read=30)
    renderer_session = aiohttp.ClientSession(timeout=timeout)
    flv_muxer_music = FlvMuxer()
    flv_muxer_voice = FlvMuxer()
    await asyncio.to_thread(background_music.load)
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
    _load_saved_motion_config()
    app = web.Application(client_max_size=80 * 1024 * 1024)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/avatars", handle_avatars)
    app.router.add_post("/avatar", handle_set_avatar)
    app.router.add_get("/motion-config", handle_motion_config)
    app.router.add_put("/motion-config", handle_set_motion_config)
    app.router.add_get("/livestream.flv", handle_livestream)
    app.router.add_post("/audio", handle_audio)
    app.router.add_post("/audio-chunk", handle_audio_chunk)
    app.router.add_post("/audio-finish", handle_audio_finish)
    app.router.add_post("/listen-chunk", handle_listen_chunk)
    app.router.add_post("/listen-reset", handle_listen_reset)
    app.router.add_post("/interrupt", handle_interrupt)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    print(
        f"avtr1 gateway {HOST}:{PORT} -> {RENDERER} avatar={AVATAR_ID} bg={BG_ID}",
        flush=True,
    )
    web.run_app(app, host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
