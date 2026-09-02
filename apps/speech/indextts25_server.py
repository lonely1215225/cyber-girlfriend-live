"""Local IndexTTS-2.5 worker. Loads weights in this process only."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models" / "indextts2.5"
DEFAULT_REPO = ROOT / "third_party" / "index-tts"
REF_DIR = ROOT / "run" / "indextts-refs"
HOST = "127.0.0.1"
PORT = int(os.environ.get("INDEXTTS_PORT", "18782"))
SAMPLE_RATE = 22050

LOG = logging.getLogger("indextts25")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_lock = threading.Lock()
_ready = False
_error = ""
_qwen_emo = False
_engine = None
_load_started = time.monotonic()


def _model_dir() -> Path:
    raw = os.environ.get("INDEXTTS_MODEL_DIR", str(DEFAULT_MODEL_DIR))
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _repo_dir() -> Path:
    raw = os.environ.get("INDEXTTS_REPO", str(DEFAULT_REPO))
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _bool_env(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip() not in {"0", "false", "False", ""}


def _vram_mb() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.memory_allocated(0) / (1024 * 1024))
    except Exception:
        return None


def _qwen_init_factory(module: Any):
    def pinned(self, model_dir):  # noqa: ANN001
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True
        )
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.float16,
            "local_files_only": True,
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": "cuda:0"}
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **kwargs)
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy",
            "愤怒": "angry",
            "悲伤": "sad",
            "恐惧": "afraid",
            "反感": "disgusted",
            "低落": "melancholic",
            "惊讶": "surprised",
            "自然": "calm",
        }
        self.desired_vector_order = [
            "高兴",
            "愤怒",
            "悲伤",
            "恐惧",
            "反感",
            "低落",
            "惊讶",
            "自然",
        ]
        self.melancholic_words = {
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
        self.max_score = 1.2
        self.min_score = 0.0

    return pinned


def _tensor_to_pcm16(chunk: Any) -> bytes:
    if chunk is None or isinstance(chunk, (list, tuple)):
        # 2.5 yields the wav list when interval_silence=0; do not replay it.
        return b""
    if torch.is_tensor(chunk):
        array = chunk.detach().cpu().float().numpy()
    else:
        array = np.asarray(chunk)
    pcm = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
    if pcm.size == 0:
        return b""
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    if peak > 2.0:
        pcm = np.clip(pcm, -32767.0, 32767.0)
    else:
        pcm = np.clip(pcm * 32767.0, -32767.0, 32767.0)
    return pcm.astype(np.int16).tobytes()


def _store_upload(prefix: str, payload: bytes, suffix: str = ".wav") -> str:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()[:20]
    path = REF_DIR / f"{prefix}-{digest}{suffix}"
    if not path.is_file():
        path.write_bytes(payload)
    return str(path)


def _patch_transformers_for_index() -> None:
    """Index vendors a transformers 4.52 generation copy; keep s2s on 4.57."""

    import torch.nn as nn
    import transformers.cache_utils as cache_utils
    import transformers.generation.candidate_generator as cand
    import transformers.generation.configuration_utils as gen_cfg
    import transformers.modeling_utils as modeling_utils

    if not hasattr(gen_cfg.GenerationConfig, "forced_decoder_ids"):
        gen_cfg.GenerationConfig.forced_decoder_ids = None

    if not hasattr(cache_utils, "QuantizedCacheConfig"):
        class QuantizedCacheConfig:
            def __init__(self, backend: str = "quanto", **kwargs):
                self.backend = backend

        cache_utils.QuantizedCacheConfig = QuantizedCacheConfig
    if not hasattr(gen_cfg, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
        gen_cfg.NEED_SETUP_CACHE_CLASSES_MAPPING = {}
    if not hasattr(gen_cfg, "QUANT_BACKEND_CLASSES_MAPPING"):
        gen_cfg.QUANT_BACKEND_CLASSES_MAPPING = {}
    if not hasattr(cand, "_crop_past_key_values"):
        def _crop_past_key_values(model, past_key_values, maximum_length):  # noqa: ARG001
            return past_key_values

        cand._crop_past_key_values = _crop_past_key_values
    if not hasattr(modeling_utils, "SequenceSummary"):
        class SequenceSummary(nn.Module):
            def __init__(self, config):  # noqa: ARG002
                super().__init__()

            def forward(self, hidden_states, cls_index=None):  # noqa: ARG002
                return hidden_states[:, 0]

        modeling_utils.SequenceSummary = SequenceSummary


def _load_engine() -> None:
    global _ready, _error, _qwen_emo, _engine
    repo = _repo_dir()
    model_dir = _model_dir()
    cfg_path = model_dir / "config.yaml"
    if not cfg_path.is_file():
        _error = f"missing {cfg_path}"
        LOG.error("%s", _error)
        return
    sys.path.insert(0, str(repo))
    cache_dir = model_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    want_qwen = _bool_env("INDEXTTS_USE_QWEN_EMO", "1")
    use_bf16 = _bool_env("INDEXTTS_USE_BF16", "1")
    try:
        _patch_transformers_for_index()
        import indextts.infer_v2_5 as infer_mod

        os.environ["HF_HUB_CACHE"] = str(cache_dir)
        infer_mod.QwenEmotion.__init__ = _qwen_init_factory(infer_mod)
        from indextts.utils.model_download import ensure_models_available

        ensure_models_available(str(model_dir))
    except Exception as exc:
        _error = f"import failed: {exc}"
        LOG.exception("IndexTTS import failed")
        return

    def construct(use_qwen: bool):
        return infer_mod.IndexTTS2(
            cfg_path=str(cfg_path),
            model_dir=str(model_dir),
            use_bf16=use_bf16,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_cuda_kernel=False,
            use_deepspeed=False,
            use_qwen_emo=use_qwen,
        )

    try:
        engine = construct(want_qwen)
        _qwen_emo = want_qwen
    except torch.cuda.OutOfMemoryError:
        LOG.warning("IndexTTS OOM with QwenEmotion; retrying vector-only")
        torch.cuda.empty_cache()
        try:
            engine = construct(False)
            _qwen_emo = False
        except Exception as exc:
            _error = f"load failed after Qwen fallback: {exc}"
            LOG.exception("IndexTTS load failed")
            return
    except Exception as exc:
        _error = f"load failed: {exc}"
        LOG.exception("IndexTTS load failed")
        return

    try:
        _warmup(engine)
    except Exception as exc:
        _error = f"warmup failed: {exc}"
        LOG.exception("IndexTTS warmup failed")
        return
    _engine = engine
    _ready = True
    _error = ""
    LOG.info(
        "IndexTTS-2.5 ready qwen_emo=%s vram_mb=%s elapsed=%.1fs",
        _qwen_emo,
        _vram_mb(),
        time.monotonic() - _load_started,
    )


def _warmup(engine: Any) -> None:
    ref = Path(os.environ.get("INDEXTTS_WARMUP_REF", "")).expanduser()
    if not ref.is_file():
        candidate = ROOT / "data" / "voices" / "system-default.wav"
        ref = candidate if candidate.is_file() else ref
    if not ref.is_file():
        LOG.info("IndexTTS warmup skipped; no reference wav")
        return
    LOG.info("IndexTTS warmup ref=%s", ref)
    gen = engine.infer(
        spk_audio_prompt=str(ref),
        text="嗯，我在。",
        output_path=None,
        lang="ZH",
        emo_vector=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        emo_alpha=0.2,
        use_emo_text=False,
        use_random=False,
        interval_silence=0,
        verbose=False,
        stream_return=True,
        duration_factor=1.0,
    )
    for _ in gen:
        pass


def _iter_pcm(
    *,
    text: str,
    lang: str,
    ref_path: str,
    emo_vector: list[float] | None,
    emo_alpha: float,
    use_emo_text: bool,
    emo_text: str,
    duration_factor: float,
    emo_audio: str | None,
    interval_silence: int,
) -> Iterator[bytes]:
    if _engine is None:
        raise RuntimeError("IndexTTS is not loaded")
    kwargs: dict[str, Any] = {
        "spk_audio_prompt": ref_path,
        "text": text,
        "output_path": None,
        "lang": lang,
        "emo_alpha": emo_alpha,
        "use_emo_text": bool(use_emo_text) and _qwen_emo,
        "use_random": False,
        "interval_silence": int(interval_silence),
        "verbose": False,
        "stream_return": True,
        "duration_factor": duration_factor,
    }
    if emo_vector:
        kwargs["emo_vector"] = emo_vector
    if emo_text and kwargs["use_emo_text"]:
        kwargs["emo_text"] = emo_text
    if emo_audio:
        kwargs["emo_audio_prompt"] = emo_audio
    started = time.monotonic()
    first = True
    for chunk in _engine.infer(**kwargs):
        payload = _tensor_to_pcm16(chunk)
        if not payload:
            continue
        if first:
            LOG.info(
                "IndexTTS first packet after %.2fs bytes=%s text=%r",
                time.monotonic() - started,
                len(payload),
                text[:40],
            )
            first = False
        yield payload


app = FastAPI(title="indextts25")


@app.on_event("startup")
def _startup() -> None:
    thread = threading.Thread(target=_load_engine, name="indextts-load", daemon=True)
    thread.start()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "ready": _ready,
        "engine": "IndexTTS-2.5",
        "sr": SAMPLE_RATE,
        "qwen_emo": _qwen_emo,
        "vram_mb": _vram_mb(),
        "error": _error,
    }


@app.post("/v1/audio/speech/stream")
async def stream_speech(
    text: str = Form(...),
    lang: str = Form("ZH"),
    ref_path: str = Form(""),
    emo_vector: str = Form("[]"),
    emo_alpha: str = Form("0.2"),
    use_emo_text: str = Form("0"),
    emo_text: str = Form(""),
    duration_factor: str = Form("1.0"),
    interval_silence: str = Form("0"),
    reference: UploadFile | None = File(None),
    emo_audio: UploadFile | None = File(None),
):
    if not _ready or _engine is None:
        raise HTTPException(status_code=503, detail=_error or "IndexTTS is still loading")
    spoken = str(text or "").strip()
    if not spoken:
        raise HTTPException(status_code=400, detail="text is empty")
    speaker = str(ref_path or "").strip()
    if reference is not None:
        payload = await reference.read()
        if payload:
            speaker = _store_upload("spk", payload)
    if not speaker or not Path(speaker).is_file():
        raise HTTPException(status_code=400, detail="reference audio is missing")
    try:
        vector = json.loads(emo_vector or "[]")
        if vector and (not isinstance(vector, list) or len(vector) != 8):
            raise ValueError("emo_vector must be 8 floats")
        vector = [float(item) for item in vector] if vector else None
        alpha = min(1.0, max(0.0, float(emo_alpha)))
        pace = min(2.0, max(0.5, float(duration_factor)))
        silence = max(0, int(float(interval_silence)))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    emotion_path = None
    if emo_audio is not None:
        blob = await emo_audio.read()
        if blob:
            emotion_path = _store_upload("emo", blob)
    if not _lock.acquire(blocking=False):
        LOG.info("IndexTTS busy; infer lock is held text=%r", spoken[:40])
        return JSONResponse(
            status_code=429,
            content={"detail": "IndexTTS is busy"},
            headers={"Retry-After": "1"},
        )
    LOG.info("IndexTTS infer lock acquired text=%r", spoken[:40])

    def generate() -> Iterator[bytes]:
        try:
            yield from _iter_pcm(
                text=spoken,
                lang=str(lang or "ZH"),
                ref_path=speaker,
                emo_vector=vector,
                emo_alpha=alpha,
                use_emo_text=use_emo_text.strip() not in {"0", "false", "False", ""},
                emo_text=str(emo_text or "").strip(),
                duration_factor=pace,
                emo_audio=emotion_path,
                interval_silence=silence,
            )
        finally:
            _lock.release()
            LOG.info("IndexTTS infer lock released text=%r", spoken[:40])

    return StreamingResponse(
        generate(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(SAMPLE_RATE)},
    )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
