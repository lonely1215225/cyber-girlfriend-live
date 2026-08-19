#!/usr/bin/env bash
# One-click install for a new GPU box. Downloads models, builds TensorRT engines,
# writes config.env. After this finishes: ./scripts/start.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

say() { echo "▸ $*"; }
die() { echo "✗ $*" >&2; exit 1; }

[[ "$(id -u)" -eq 0 ]] || die "请用 root 运行：sudo ./install.sh"
command -v nvidia-smi >/dev/null || die "需要 NVIDIA GPU 和 nvidia-smi"
nvidia-smi >/dev/null || die "GPU 驱动不可用"

LISTEN_HTTP_PORT="${LISTEN_HTTP_PORT:-19800}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME" "$ROOT/models" "$ROOT/logs" "$ROOT/run" "$ROOT/proxy/certs"

if [[ -f /workspace/.hf_token ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '[:space:]' < /workspace/.hf_token)"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

say "[1/7] 系统依赖"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential cmake git curl ffmpeg libsndfile1 pkg-config \
  nginx openssl ca-certificates python3 python3-venv python3-pip unzip iproute2 procps

PUBLIC_IP="${PUBLIC_IP:-$(curl -4 -fsS --max-time 8 ifconfig.me 2>/dev/null || true)}"
PUBLIC_IP="${PUBLIC_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
PUBLIC_IP="${PUBLIC_IP:-127.0.0.1}"
say "PUBLIC_IP=$PUBLIC_IP"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || die "uv 安装失败"

if ! command -v pixi >/dev/null; then
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="$HOME/.pixi/bin:$PATH"
fi
export PATH="$HOME/.pixi/bin:$HOME/.local/bin:$PATH"
command -v pixi >/dev/null || die "pixi 安装失败"

say "[2/7] Ollama + LLM"
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
    sleep 1
  done
fi
curl -sf http://127.0.0.1:11434/api/tags >/dev/null || die "Ollama 未启动"
LLM_NAME="${LLM_NAME:-jaahas/qwen3.5-uncensored:9b}"
ollama pull "$LLM_NAME"

say "[3/7] 语音环境 speech-to-speech"
S2S_VENV="$ROOT/s2s/.venv"
uv python install 3.12
[[ -x "$S2S_VENV/bin/python" ]] || uv venv --python 3.12 "$S2S_VENV"
uv pip install --python "$S2S_VENV/bin/python" -r "$ROOT/s2s/hf-realtime-voice/requirements.txt"
uv pip install --python "$S2S_VENV/bin/python" "speech-to-speech[faster-whisper]==0.2.10" "faster-whisper==1.2.1" \
  "faster-qwen3-tts==0.3.2" "qwen-tts==0.1.1" "transformers==4.57.3" \
  "torch==2.6.0" "torchaudio==2.6.0" \
  "numpy<2.3" "numba>=0.61" "nvidia-cudnn-cu12>=9,<10" nvidia-cublas-cu12 hf_transfer
[[ -x "$S2S_VENV/bin/speech-to-speech" ]] || die "speech-to-speech 安装失败"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
mkdir -p "$TORCH_HOME"
"$S2S_VENV/bin/python" - <<'PY'
import nltk
import torch
from huggingface_hub import snapshot_download
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
snapshot_download("Systran/faster-whisper-large-v3")
torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True, skip_validation=True)
PY

say "[4/7] 下载 Qwen3-TTS（约 8GB）"
TTS_DIR="$ROOT/models/qwen3tts/Qwen3-TTS-12Hz-1.7B-Base"
if [[ ! -f "$TTS_DIR/config.json" ]]; then
  "$S2S_VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_dir="$TTS_DIR")
PY
fi

say "[5/7] AVTR-1 环境 + 权重"
AVTR1="$ROOT/third_party/avtr-1"
export AVTR1_LOCAL_STORAGE="$AVTR1/artifacts"
AVTR1_HF_HOME="$ROOT/.cache/avtr1-huggingface"
# pixi.toml asks for CUDA 12.8; driver CUDA is often 12.x. Override so install
# uses the lockfile instead of failing the virtual-package check.
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12.8}"
(
  cd "$AVTR1"
  # --frozen: do not re-solve. A toml/lock mismatch previously made pixi run hang.
  pixi install --frozen -e renderer
)
AVTR1_PY=""
# Keep this order in sync with scripts/start.sh. `renderer` is the canonical
# environment declared in pixi.toml; `default` only supports older installs.
for cand in renderer default; do
  if [[ -x "$AVTR1/.pixi/envs/$cand/bin/python" ]]; then
    AVTR1_PY="$AVTR1/.pixi/envs/$cand/bin/python"
    break
  fi
done
[[ -x "$AVTR1_PY" ]] || die "pixi 环境未生成 python"
# Pixi envs typically have no `pip` binary; use uv against that interpreter.
"$AVTR1_PY" -c "import aiohttp" 2>/dev/null \
  || uv pip install --python "$AVTR1_PY" "aiohttp>=3.10,<4"
"$AVTR1_PY" -c "import aiohttp" || die "aiohttp 未装上，网关无法启动"
HF_HOME="$AVTR1_HF_HOME" "$AVTR1_PY" "$AVTR1/scripts/download_artifacts.py"

say "[6/7] 写入默认形象并编译 TensorRT（首次较久）"
LOOKS=(xiaoya xiaoya_beach_close xiaoya_beach xiaoya_locket)
FRAMES="$AVTR1/artifacts/main/avatars_artifacts/reference_frames"
BGS="$AVTR1/artifacts/main/avatars_artifacts/backgrounds"
mkdir -p "$FRAMES" "$BGS" "$AVTR1/artifacts/main/renderer_runtime_artifacts"
PLUGIN_SRC="$ROOT/assets/libgrid_sample_3d_plugin.so"
[[ -f "$PLUGIN_SRC" ]] || die "缺少 TensorRT 插件 $PLUGIN_SRC"
cp -f "$PLUGIN_SRC" \
  "$AVTR1/artifacts/main/renderer_runtime_artifacts/libgrid_sample_3d_plugin.so"
for id in "${LOOKS[@]}"; do
  [[ -f "$ROOT/assets/looks/${id}.png" ]] || die "缺少形象 $ROOT/assets/looks/${id}.png"
  cp -f "$ROOT/assets/looks/${id}.png" "$FRAMES/${id}.png"
done
[[ -f "$ROOT/assets/looks/plain_white.png" ]] && cp -f "$ROOT/assets/looks/plain_white.png" "$BGS/plain_white.png"
[[ -f "$ROOT/assets/looks/pasteback_mask.png" ]] && cp -f "$ROOT/assets/looks/pasteback_mask.png" \
  "$AVTR1/artifacts/main/avatars_artifacts/pasteback_mask.png"
(
  cd "$AVTR1"
  HF_HOME="$AVTR1_HF_HOME" pixi run --frozen -e renderer build-trt-engines
)
ENGINE="$AVTR1/artifacts/main/speech2motion_runtime_artifacts_cc/avtr1_decode_fp16.engine"
[[ -f "$ENGINE" ]] || die "TRT 引擎未生成：$ENGINE"
# ONNX/TorchScript sources and their dedicated HF cache are only inputs to the
# TensorRT build. Keeping them after all engines exist wastes roughly 5 GB.
for generated_dir in "$AVTR1/artifacts/main/build_artifacts" "$AVTR1_HF_HOME"; do
  [[ ! -d "$generated_dir" ]] || find "$generated_dir" -depth -delete
done

say "[7/7] 写 config.env 和证书"
if [[ ! -f "$ROOT/config.env" ]]; then
  sed -e "s|^PUBLIC_IP=.*|PUBLIC_IP=$PUBLIC_IP|" \
      -e "s|^LISTEN_HTTP_PORT=.*|LISTEN_HTTP_PORT=$LISTEN_HTTP_PORT|" \
      -e "s|^PUBLIC_HTTP_PORT=.*|PUBLIC_HTTP_PORT=$LISTEN_HTTP_PORT|" \
      -e "s|^PUBLIC_WS_URL=.*|PUBLIC_WS_URL=wss://${PUBLIC_IP}:${LISTEN_HTTP_PORT}/v1/realtime|" \
      -e "s|^TTS_MODEL=.*|TTS_MODEL=$TTS_DIR|" \
      -e "s|^REF_AUDIO=.*|REF_AUDIO=$ROOT/assets/ref16k.wav|" \
      "$ROOT/config.env.example" > "$ROOT/config.env"
else
  say "已有 config.env，未覆盖"
fi
if [[ ! -f "$ROOT/proxy/certs/server.crt" ]]; then
  if [[ "$PUBLIC_IP" =~ ^[0-9a-fA-F:.]+$ ]]; then
    SAN="IP:${PUBLIC_IP}"
  else
    SAN="DNS:${PUBLIC_IP}"
  fi
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$ROOT/proxy/certs/server.key" \
    -out "$ROOT/proxy/certs/server.crt" \
    -days 3650 \
    -subj "/CN=${PUBLIC_IP}" \
    -addext "subjectAltName=${SAN}"
fi

cat <<TXT

安装完成。

  配置: $ROOT/config.env
  公网 IP: $PUBLIC_IP
  启动: $ROOT/scripts/start.sh
  停止: $ROOT/scripts/stop.sh

  浏览器打开: https://${PUBLIC_IP}:${LISTEN_HTTP_PORT}/
  首次请信任自签证书，并允许麦克风。

TXT
