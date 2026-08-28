#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/config.env"

RUN="$ROOT/run"
LOG="$ROOT/logs"
mkdir -p "$RUN" "$LOG"

S2S_VENV="$ROOT/s2s/.venv"
FRONTEND="$ROOT/s2s/hf-realtime-voice"

# Allow relative paths in config.env
[[ "${TTS_MODEL:-}" = /* ]] || TTS_MODEL="$ROOT/${TTS_MODEL:-models/qwen3tts/Qwen3-TTS-12Hz-1.7B-Base}"
[[ "${REF_AUDIO:-}" = /* ]] || REF_AUDIO="$ROOT/${REF_AUDIO:-assets/ref16k.wav}"
export TTS_MODEL REF_AUDIO REF_TEXT
SENSEVOICE_MODEL="${SENSEVOICE_MODEL:-models/sensevoice/SenseVoiceSmall}"
if [[ "$SENSEVOICE_MODEL" == models/* ]]; then
  SENSEVOICE_MODEL="$ROOT/$SENSEVOICE_MODEL"
fi

say() { echo "▸ $*"; }
die() { echo "✗ $*" >&2; exit 1; }

alive() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] || return 1
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_bg() {
  local name="$1" pidfile="$2" logfile="$3"
  shift 3
  if alive "$pidfile"; then
    say "$name already running (pid $(cat "$pidfile"))"
    return 0
  fi
  # A stale pidfile must never make a subsequent one-click start look healthy.
  rm -f "$pidfile"
  say "start $name"
  # Detach from the caller's session as well as its terminal. This keeps
  # services alive when start.sh is launched by SSH or another job runner.
  nohup setsid "$@" >"$logfile" 2>&1 &
  echo $! >"$pidfile"
  sleep 0.4
  alive "$pidfile" || die "$name exited immediately, see $logfile"
}

wait_port() {
  local port="$1" label="$2" tries="${3:-90}"
  local i
  for i in $(seq 1 "$tries"); do
    if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
      say "$label ready on :$port"
      return 0
    fi
    sleep 2
  done
  die "$label did not listen on :$port"
}

[[ -x "$S2S_VENV/bin/speech-to-speech" ]] || die "speech-to-speech is not installed；先运行 ./install.sh"
[[ -s "$TTS_MODEL/config.json" ]] || die "Qwen3-TTS weight is missing at $TTS_MODEL；先运行 ./install.sh"
[[ -d "$FRONTEND" ]] || die "frontend is missing"
[[ -f "$REF_AUDIO" ]] || die "missing ref audio $REF_AUDIO"
mkdir -p "$FRONTEND/avatar"
mkdir -p "$ROOT/proxy/certs"
if [[ ! -f "$ROOT/proxy/certs/server.crt" ]]; then
  if [[ "${PUBLIC_IP}" =~ ^[0-9a-fA-F:.]+$ ]]; then
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

if ! curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  if command -v systemctl >/dev/null && systemctl start ollama 2>/dev/null; then
    say "started ollama via systemd"
  elif command -v ollama >/dev/null; then
    say "starting ollama serve"
    nohup ollama serve >"$LOG/ollama.log" 2>&1 &
  fi
  for _ in $(seq 1 30); do
    curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
fi
curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1 || die "Ollama is not reachable at $OLLAMA_URL"

# Load the configured model now instead of making the first viewer wait for
# Ollama to allocate GPU memory. The stop script unloads this exact model while
# deliberately leaving a possibly shared Ollama daemon running.
if [[ "${SKIP_LLM_PREWARM:-0}" != "1" && "${LLM_PREWARM:-1}" != "0" ]]; then
  [[ "$LLM_NAME" =~ ^[A-Za-z0-9._:/-]+$ ]] || die "LLM_NAME contains unsupported characters"
  [[ "${LLM_NUM_CTX:-4096}" =~ ^[0-9]+$ ]] || die "LLM_NUM_CTX must be an integer"
  [[ "${LLM_KEEP_ALIVE:--1}" =~ ^-?[0-9]+$ ]] || die "LLM_KEEP_ALIVE must be an integer"
  say "prewarm Ollama model $LLM_NAME"
  OLLAMA_PREWARM_PAYLOAD="$(printf \
    '{"model":"%s","prompt":"","stream":false,"keep_alive":%s,"options":{"num_ctx":%s,"num_predict":1}}' \
    "$LLM_NAME" "${LLM_KEEP_ALIVE:--1}" "${LLM_NUM_CTX:-4096}")"
  curl -fsS --max-time 600 \
    -H 'Content-Type: application/json' \
    -d "$OLLAMA_PREWARM_PAYLOAD" \
    "$OLLAMA_URL/api/generate" >/dev/null \
    || die "failed to prewarm Ollama model $LLM_NAME"
  say "Ollama model ready"
fi

# A tiny resident model only produces the first fact-free conversational
# bridge. It runs concurrently with Grok and must be warm before a viewer asks
# the first question; otherwise model loading would erase the latency win.
local_lead_model="${LOCAL_LEAD_MODEL:-jaahas/qwen3.5-uncensored:9b}"
if [[ "${LOCAL_LEAD_ENABLED:-1}" == "1" && -n "$local_lead_model" \
      && "$local_lead_model" != "$LLM_NAME" ]]; then
  [[ "$local_lead_model" =~ ^[A-Za-z0-9._:/-]+$ ]] \
    || die "LOCAL_LEAD_MODEL contains unsupported characters"
  say "prewarm local lead model $local_lead_model"
  LOCAL_LEAD_PREWARM_PAYLOAD="$(printf \
    '{"model":"%s","prompt":"","stream":false,"keep_alive":-1,"options":{"num_ctx":2048,"num_predict":1}}' \
    "$local_lead_model")"
  curl -fsS --max-time 180 \
    -H 'Content-Type: application/json' \
    -d "$LOCAL_LEAD_PREWARM_PAYLOAD" \
    "$OLLAMA_URL/api/generate" >/dev/null \
    || die "failed to prewarm local lead model $local_lead_model"
  say "local lead model ready"
fi

S2S_NVIDIA_LIBS="$("$S2S_VENV/bin/python" - <<'PY' 2>/dev/null || true
import os
import pkgutil
try:
    import nvidia
except Exception:
    raise SystemExit
paths = []
for mod in pkgutil.iter_modules(nvidia.__path__):
    lib = os.path.join(nvidia.__path__[0], mod.name, "lib")
    if os.path.isdir(lib):
        paths.append(lib)
print(":".join(paths))
PY
)"
S2S_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ -n "${S2S_NVIDIA_LIBS:-}" ]]; then
  S2S_LD_LIBRARY_PATH="$S2S_NVIDIA_LIBS${S2S_LD_LIBRARY_PATH:+:$S2S_LD_LIBRARY_PATH}"
fi
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/.cache/modelscope}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MODELSCOPE_CACHE"
unset HF_HUB_ENABLE_HF_TRANSFER
if [[ -f /workspace/.hf_token ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '[:space:]' < /workspace/.hf_token)"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
export PUBLIC_WS_URL="${PUBLIC_WS_URL:-wss://${PUBLIC_IP}:${PUBLIC_HTTP_PORT}/v1/realtime}"
export PUBLIC_IP PUBLIC_HTTP_PORT
export AVATAR_SAMPLE_RATE
export SPEECH_TO_SPEECH_URL=auto
export LIVE_ROOM_ENABLED="${LIVE_ROOM_ENABLED:-1}"
export LIVE_ROOM_QUEUE_LIMIT="${LIVE_ROOM_QUEUE_LIMIT:-100}"
export LIVE_ROOM_JOIN_TIMEOUT="${LIVE_ROOM_JOIN_TIMEOUT:-60}"
export LIVE_ROOM_MAX_CALL_SECONDS="${LIVE_ROOM_MAX_CALL_SECONDS:-600}"
export ADMIN_SETTINGS_PASSWORD="${ADMIN_SETTINGS_PASSWORD:-123456}"
export ADMIN_SESSION_TTL_SECONDS="${ADMIN_SESSION_TTL_SECONDS:-1800}"
ROOM_DB_PATH="${ROOM_DB_PATH:-data/live_room.sqlite3}"
[[ "$ROOM_DB_PATH" = /* ]] || ROOM_DB_PATH="$ROOT/$ROOM_DB_PATH"
export ROOM_DB_PATH
export ROOM_IP_RETENTION_DAYS="${ROOM_IP_RETENTION_DAYS:-30}"
export MENTION_REPLY_QUEUE_LIMIT="${MENTION_REPLY_QUEUE_LIMIT:-30}"
export BACKGROUND_MUSIC_ENABLED="${BACKGROUND_MUSIC_ENABLED:-1}"
BACKGROUND_MUSIC_DIR="${BACKGROUND_MUSIC_DIR:-.}"
[[ "$BACKGROUND_MUSIC_DIR" = /* ]] || BACKGROUND_MUSIC_DIR="$ROOT/$BACKGROUND_MUSIC_DIR"
export BACKGROUND_MUSIC_DIR
export BACKGROUND_MUSIC_VOLUME="${BACKGROUND_MUSIC_VOLUME:-0.16}"
export BACKGROUND_MUSIC_DUCK_VOLUME="${BACKGROUND_MUSIC_DUCK_VOLUME:-0.04}"
export BACKGROUND_MUSIC_USER_RMS="${BACKGROUND_MUSIC_USER_RMS:-450}"
export MCP_ENABLED="${MCP_ENABLED:-1}"
export MCP_COINGECKO_URL="${MCP_COINGECKO_URL:-https://mcp.api.coingecko.com/mcp}"
export MCP_EXA_URL="${MCP_EXA_URL:-https://mcp.exa.ai/mcp}"
export MCP_GDELT_URL="${MCP_GDELT_URL:-https://gdelt.caseyjhand.com/mcp}"
export MCP_TAVILY_URL="${MCP_TAVILY_URL:-}"
export MCP_MAX_OUTPUT_CHARS="${MCP_MAX_OUTPUT_CHARS:-6000}"
export TAVILY_API_KEY="${TAVILY_API_KEY:-}"
export EXA_API_KEY="${EXA_API_KEY:-}"
export JINA_API_KEY="${JINA_API_KEY:-}"
export SEARXNG_URL="${SEARXNG_URL:-}"
export JINA_READER_ENABLED="${JINA_READER_ENABLED:-1}"
export SMART_SEARCH_TIMEOUT_SECONDS="${SMART_SEARCH_TIMEOUT_SECONDS:-5}"
export SMART_SEARCH_EVIDENCE_BUDGET_SECONDS="${SMART_SEARCH_EVIDENCE_BUDGET_SECONDS:-3.5}"
export SMART_SEARCH_CACHE_SECONDS="${SMART_SEARCH_CACHE_SECONDS:-180}"
export SMART_SEARCH_COOLDOWN_SECONDS="${SMART_SEARCH_COOLDOWN_SECONDS:-60}"
export NEWS_RSS_ENABLED="${NEWS_RSS_ENABLED:-1}"
export NEWS_GOOGLE_RSS_ENABLED="${NEWS_GOOGLE_RSS_ENABLED:-1}"
export NEWS_RSS_TIMEOUT="${NEWS_RSS_TIMEOUT:-8}"
export NEWS_RSS_CACHE_SECONDS="${NEWS_RSS_CACHE_SECONDS:-120}"
export NEWS_RSS_MAX_ITEMS="${NEWS_RSS_MAX_ITEMS:-10}"
export NEWS_RSS_MAX_AGE_HOURS="${NEWS_RSS_MAX_AGE_HOURS:-72}"
export NEWS_RSS_FEEDS="${NEWS_RSS_FEEDS:-}"
export NEWS_RSS_ALLOWED_HOSTS="${NEWS_RSS_ALLOWED_HOSTS:-}"
export S2S_INTERNAL_WS_URL="ws://127.0.0.1:${S2S_PORT}/v1/realtime"
export TTS_EMOTION_ENABLED="${TTS_EMOTION_ENABLED:-1}"
export TTS_STYLE_INSTRUCT_ENABLED="${TTS_STYLE_INSTRUCT_ENABLED:-1}"
export TTS_TEMPERATURE="${TTS_TEMPERATURE:-0.65}"
export TTS_TOP_K="${TTS_TOP_K:-30}"
export TTS_TOP_P="${TTS_TOP_P:-0.85}"
export TTS_DO_SAMPLE="${TTS_DO_SAMPLE:-1}"
export TTS_REPETITION_PENALTY="${TTS_REPETITION_PENALTY:-1.05}"
export AGENT_FOCUS_TTL_SECONDS="${AGENT_FOCUS_TTL_SECONDS:-1800}"
export AGENT_PROACTIVE_COOLDOWN_SECONDS="${AGENT_PROACTIVE_COOLDOWN_SECONDS:-60}"
export AGENT_TIMEZONE="${AGENT_TIMEZONE:-Asia/Shanghai}"
export AVATAR_TEE_UPLOAD_PREROLL_MS="${AVATAR_TEE_UPLOAD_PREROLL_MS:-320}"
export AVATAR_TEE_SEGMENT_GAP_MS="${AVATAR_TEE_SEGMENT_GAP_MS:-1200}"
export AVTR1_SPEECH_START_BUFFER_MS="${AVTR1_SPEECH_START_BUFFER_MS:-600}"
export AVTR1_AUDIO_REBUFFER_STEP_MS="${AVTR1_AUDIO_REBUFFER_STEP_MS:-200}"
export AVTR1_AUDIO_MAX_BUFFER_MS="${AVTR1_AUDIO_MAX_BUFFER_MS:-1400}"
export AVTR1_OUTPUT_RESERVOIR_MS="${AVTR1_OUTPUT_RESERVOIR_MS:-800}"
export AVTR1_PROACTIVE_OUTPUT_RESERVOIR_MS="${AVTR1_PROACTIVE_OUTPUT_RESERVOIR_MS:-1200}"
export AVTR1_VIDEO_ENCODE_QUEUE_FRAMES="${AVTR1_VIDEO_ENCODE_QUEUE_FRAMES:-8}"
export AVTR1_MAX_SPEECH_SECONDS="${AVTR1_MAX_SPEECH_SECONDS:-90}"
export WEBRTC_ENABLED="${WEBRTC_ENABLED:-1}"
export WEBRTC_PUBLIC_HOST="${WEBRTC_PUBLIC_HOST:-${PUBLIC_IP}}"
export WEBRTC_UDP_PORT="${WEBRTC_UDP_PORT:-8189}"
export WEBRTC_TCP_PORT="${WEBRTC_TCP_PORT:-8190}"
export MEDIAMTX_WHEP_PORT="${MEDIAMTX_WHEP_PORT:-18889}"
export MEDIAMTX_RTSP_PORT="${MEDIAMTX_RTSP_PORT:-18554}"
export MEDIAMTX_API_PORT="${MEDIAMTX_API_PORT:-19997}"
export MEDIAMTX_METRICS_PORT="${MEDIAMTX_METRICS_PORT:-19998}"
export WEBRTC_OPUS_BITRATE="${WEBRTC_OPUS_BITRATE:-48000}"
export WEBRTC_PACKET_LOSS_PERCENT="${WEBRTC_PACKET_LOSS_PERCENT:-5}"
export STARTUP_GREETING="${STARTUP_GREETING:-}"
export IDLE_PROMPT="${IDLE_PROMPT:-}"
export IDLE_PROMPT_MIN_SECONDS="${IDLE_PROMPT_MIN_SECONDS:-35}"
export IDLE_PROMPT_MAX_SECONDS="${IDLE_PROMPT_MAX_SECONDS:-55}"
export PROACTIVE_NEWS_MIN_SECONDS="${PROACTIVE_NEWS_MIN_SECONDS:-90}"
export PROACTIVE_NEWS_MAX_SECONDS="${PROACTIVE_NEWS_MAX_SECONDS:-150}"
export GROK_ENABLED="${GROK_ENABLED:-0}"
export GROK_PROXY_PORT="${GROK_PROXY_PORT:-18080}"
export GROK_PROXY_BASE_URL="${GROK_PROXY_BASE_URL:-http://127.0.0.1:${GROK_PROXY_PORT}/v1}"
export GROK_MODEL="${GROK_MODEL:-grok-4.6}"
export GROK_FAST_MODEL="${GROK_FAST_MODEL:-grok-4.5}"
export GROK_REASONING_EFFORT="${GROK_REASONING_EFFORT:-low}"
export GROK_TIMEOUT_SECONDS="${GROK_TIMEOUT_SECONDS:-45}"
export GROK_MAX_CONCURRENCY_PER_ACCOUNT="${GROK_MAX_CONCURRENCY_PER_ACCOUNT:-2}"
export LOCAL_LEAD_ENABLED="${LOCAL_LEAD_ENABLED:-0}"
export LOCAL_LEAD_MODEL="${LOCAL_LEAD_MODEL:-jaahas/qwen3.5-uncensored:9b}"
export LOCAL_LEAD_TIMEOUT_SECONDS="${LOCAL_LEAD_TIMEOUT_SECONDS:-1.4}"
export LOCAL_LEAD_MAX_CHARS="${LOCAL_LEAD_MAX_CHARS:-24}"
export OLLAMA_URL LLM_NAME THINKLESS_PORT LLM_NUM_CTX LLM_NUM_PREDICT LLM_KEEP_ALIVE LLM_PREWARM
export LLM_LOCAL_READ_TIMEOUT_SECONDS="${LLM_LOCAL_READ_TIMEOUT_SECONDS:-12}"
export LLM_CHAT_SIZE="${LLM_CHAT_SIZE:-12}"
export LLM_STREAM_BATCH_SENTENCES="${LLM_STREAM_BATCH_SENTENCES:-1}"
export LLM_COMPACTION_NUM_PREDICT="${LLM_COMPACTION_NUM_PREDICT:-256}"
export LLM_COMPACTION_MODE="${LLM_COMPACTION_MODE:-local}"
export LLM_COMPACTION_MAX_CHARS="${LLM_COMPACTION_MAX_CHARS:-900}"
export MEMORY_SEMANTIC_ENABLED="${MEMORY_SEMANTIC_ENABLED:-1}"
export MEMORY_SEMANTIC_IDLE_SECONDS="${MEMORY_SEMANTIC_IDLE_SECONDS:-12}"
export MEMORY_SEMANTIC_MAX_SECONDS="${MEMORY_SEMANTIC_MAX_SECONDS:-15}"
export MEMORY_SEMANTIC_NUM_PREDICT="${MEMORY_SEMANTIC_NUM_PREDICT:-256}"

# Keep every service log bounded while preserving one previous segment.
start_bg log_guard "$RUN/log_guard.pid" "$LOG/log_guard.log" \
  bash "$ROOT/scripts/log_guard.sh" "$LOG"

# Grok stays private on loopback.  The official CLI owns the OAuth file; the
# proxy only reads it and persists refreshed credentials in an owner-only
# runtime directory outside the Git checkout.
if [[ "$GROK_ENABLED" == "1" ]]; then
  [[ -x /usr/local/bin/grok-reverse-proxy ]] || die "Grok proxy binary is missing"
  [[ -f /root/.grok/auth.json ]] || die "Grok is not logged in; run: grok login --device-auth"
  mkdir -p /root/.local/share/grok-reverse-proxy
  chmod 700 /root/.local/share/grok-reverse-proxy
  start_bg grok_proxy "$RUN/grok_proxy.pid" "$LOG/grok_proxy.log" \
    env GROK_LISTEN_ADDR="127.0.0.1:${GROK_PROXY_PORT}" \
      GROK_AUTH_FILES=/root/.grok/auth.json \
      GROK_STATE_FILE=/root/.local/share/grok-reverse-proxy/accounts.json \
      GROK_REQUEST_TIMEOUT="${GROK_TIMEOUT_SECONDS}s" \
      GROK_MAX_CONCURRENCY_PER_ACCOUNT="$GROK_MAX_CONCURRENCY_PER_ACCOUNT" \
      /usr/local/bin/grok-reverse-proxy
  wait_port "$GROK_PROXY_PORT" "grok-proxy" 20
  curl -fsS "http://127.0.0.1:${GROK_PROXY_PORT}/healthz" >/dev/null \
    || die "Grok proxy health check failed"
fi

# 0) Ollama think=false shim (Ollama /v1/responses ignores think and stalls 20s+)
start_bg thinkless "$RUN/thinkless.pid" "$LOG/thinkless.log" \
  "$S2S_VENV/bin/python" "$ROOT/proxy/ollama_thinkless.py"
wait_port "${THINKLESS_PORT:-11435}" "ollama-thinkless" 20
if ! curl -sf "$LLM_BASE_URL/models" >/dev/null 2>&1; then
  die "thinkless proxy is not serving $LLM_BASE_URL/models"
fi

# 1) AVTR-1 HTTP-FLV
AVTR1_ROOT="$ROOT/third_party/avtr-1"
AVTR1_PY=""
# `renderer` is the canonical pixi environment. Fall back to `default` for
# deployment directories created by an older installer.
for cand in renderer default; do
  if [[ -x "$AVTR1_ROOT/.pixi/envs/$cand/bin/python" ]]; then
    AVTR1_PY="$AVTR1_ROOT/.pixi/envs/$cand/bin/python"
    break
  fi
done
[[ -x "$AVTR1_PY" ]] || die "AVTR-1 pixi env is missing；先运行 ./install.sh"
[[ -f "$AVTR1_ROOT/artifacts/main/speech2motion_runtime_artifacts_cc/avtr1_decode_fp16.engine" ]] \
  || die "AVTR-1 TRT engines missing；先运行 ./install.sh"
# Plugin was built against pixi's newer libstdc++ and TensorRT 10. System
# Ubuntu 22.04 libstdc++ only has GLIBCXX_3.4.30.
AVTR1_ENV="$(cd "$(dirname "$AVTR1_PY")/.." && pwd)"
for extra in "$AVTR1_ENV/lib" "$AVTR1_ENV/lib/python3.12/site-packages/tensorrt_libs"; do
  [[ -d "$extra" ]] && LD_LIBRARY_PATH="$extra${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH
export AVTR1_LOCAL_STORAGE="$AVTR1_ROOT/artifacts"
export AVTR1_AVATAR_ID="${AVTR1_AVATAR_ID:-xiaoya_locket}"
export AVTR1_BG_ID="${AVTR1_BG_ID:-plain_white}"
export AVTR1_OUT_H="${AVTR1_OUT_H:-1280}"
export AVTR1_OUT_W="${AVTR1_OUT_W:-720}"
export AVTR1_H264_BITRATE="${AVTR1_H264_BITRATE:-1200000}"
export AVTR1_CFG_SELF_AUDIO="${AVTR1_CFG_SELF_AUDIO:-2.3}"
export AVTR1_CFG_OTHER_AUDIO="${AVTR1_CFG_OTHER_AUDIO:-2.0}"
export AVTR1_CFG_KP="${AVTR1_CFG_KP:-3.0}"
export AVTR1_NOISE_ALPHA="${AVTR1_NOISE_ALPHA:-1.5}"
export AVTR1_NOISE_TRUNC_Z="${AVTR1_NOISE_TRUNC_Z:-1.0}"
export AVTR1_IDLE_NOISE_ALPHA="${AVTR1_IDLE_NOISE_ALPHA:-2.0}"
export AVTR1_IDLE_NOISE_TRUNC_Z="${AVTR1_IDLE_NOISE_TRUNC_Z:-1.2}"
export AVTR1_MOTION_AUDIO_RMS="${AVTR1_MOTION_AUDIO_RMS:-80}"
export AVTR1_MOTION_LISTEN_RMS="${AVTR1_MOTION_LISTEN_RMS:-450}"
export AVTR1_MOTION_ACTIVE_HOLD_SECONDS="${AVTR1_MOTION_ACTIVE_HOLD_SECONDS:-1.0}"
export AVTR1_BLINK_ENABLED="${AVTR1_BLINK_ENABLED:-${AVTR1_IDLE_BLINK_ENABLED:-1}}"
export AVTR1_BLINK_MIN_SECONDS="${AVTR1_BLINK_MIN_SECONDS:-${AVTR1_IDLE_BLINK_MIN_SECONDS:-2.4}}"
export AVTR1_BLINK_MAX_SECONDS="${AVTR1_BLINK_MAX_SECONDS:-${AVTR1_IDLE_BLINK_MAX_SECONDS:-6.8}}"
export AVTR1_BLINK_STRENGTH="${AVTR1_BLINK_STRENGTH:-${AVTR1_IDLE_BLINK_STRENGTH:-1.08}}"
export AVTR1_BLINK_SPEECH_STRENGTH="${AVTR1_BLINK_SPEECH_STRENGTH:-${AVTR1_BLINK_STRENGTH}}"
export AVTR1_BLINK_SPEECH_INTERVAL_SCALE="${AVTR1_BLINK_SPEECH_INTERVAL_SCALE:-0.82}"
export AVTR1_BLINK_DOUBLE_PROBABILITY="${AVTR1_BLINK_DOUBLE_PROBABILITY:-0.08}"
export AVTR1_BLINK_PARTIAL_PROBABILITY="${AVTR1_BLINK_PARTIAL_PROBABILITY:-0.28}"
export AVTR1_IDLE_BREATH_ENABLED="${AVTR1_IDLE_BREATH_ENABLED:-1}"
export AVTR1_IDLE_BREATH_POSE_DEGREES="${AVTR1_IDLE_BREATH_POSE_DEGREES:-0.65}"
export AVTR1_IDLE_BREATH_PITCH_RATIO="${AVTR1_IDLE_BREATH_PITCH_RATIO:-0.08}"
export AVTR1_IDLE_BREATH_YAW_RATIO="${AVTR1_IDLE_BREATH_YAW_RATIO:-1.0}"
export AVTR1_IDLE_BREATH_ROLL_RATIO="${AVTR1_IDLE_BREATH_ROLL_RATIO:--0.12}"
export AVTR1_IDLE_BREATH_PRIMARY_SECONDS="${AVTR1_IDLE_BREATH_PRIMARY_SECONDS:-4.0}"
export AVTR1_IDLE_BREATH_DRIFT_SECONDS="${AVTR1_IDLE_BREATH_DRIFT_SECONDS:-9.1}"
export AVTR1_IDLE_BREATH_DRIFT_MIX="${AVTR1_IDLE_BREATH_DRIFT_MIX:-0.30}"
export AVTR1_IDLE_BREATH_FADE_IN_STEP="${AVTR1_IDLE_BREATH_FADE_IN_STEP:-0.08}"
export AVTR1_IDLE_BREATH_FADE_OUT_STEP="${AVTR1_IDLE_BREATH_FADE_OUT_STEP:-0.18}"
export AVTR1_IDLE_EXPRESSION_ENABLED="${AVTR1_IDLE_EXPRESSION_ENABLED:-1}"
export AVTR1_IDLE_EXPRESSION_MIN_SECONDS="${AVTR1_IDLE_EXPRESSION_MIN_SECONDS:-6}"
export AVTR1_IDLE_EXPRESSION_MAX_SECONDS="${AVTR1_IDLE_EXPRESSION_MAX_SECONDS:-14}"
export AVTR1_IDLE_EXPRESSION_INTENSITY="${AVTR1_IDLE_EXPRESSION_INTENSITY:-0.58}"
export AVTR1_IDLE_EXPRESSION_QUIET_SECONDS="${AVTR1_IDLE_EXPRESSION_QUIET_SECONDS:-3.0}"
export AVTR1_URL="http://127.0.0.1:${AVTR1_PORT:-18012}"
export AVTR1_EXPRESSION_DIR="${AVTR1_EXPRESSION_DIR:-$ROOT/assets/expressions/xiaoya_locket}"
export AVTR1_EXPRESSION_RETARGET_GAIN="${AVTR1_EXPRESSION_RETARGET_GAIN:-4.0}"
export AVTR1_EXPRESSION_SOURCE_AVATAR="${AVTR1_EXPRESSION_SOURCE_AVATAR:-xiaoya_locket}"
export AVTR1_EXPRESSION_SOURCE_PREFIX="${AVTR1_EXPRESSION_SOURCE_PREFIX:-xiaoya_locket_expr_}"
export AVTR1_EXPRESSION_SOURCE_MIN_INTENSITY="${AVTR1_EXPRESSION_SOURCE_MIN_INTENSITY:-0.48}"
export AVTR1_EXPRESSION_ATTACK_FRAMES="${AVTR1_EXPRESSION_ATTACK_FRAMES:-24}"
export AVTR1_EXPRESSION_RELEASE_FRAMES="${AVTR1_EXPRESSION_RELEASE_FRAMES:-28}"
export AVTR1_EXPRESSION_ATTACK_STEP="${AVTR1_EXPRESSION_ATTACK_STEP:-0.035}"
export AVTR1_EXPRESSION_RELEASE_STEP="${AVTR1_EXPRESSION_RELEASE_STEP:-0.025}"
export AVTR1_EXPRESSION_TRANSITION_SPEECH_FRAMES="${AVTR1_EXPRESSION_TRANSITION_SPEECH_FRAMES:-5}"
export AVTR1_EXPRESSION_TRANSITION_IDLE_FRAMES="${AVTR1_EXPRESSION_TRANSITION_IDLE_FRAMES:-8}"
export AVTR1_LOCAL_TEE_URL="http://127.0.0.1:${AVATAR_GW_PORT:-18011}"
export LOAD_BALANCER_URL=disabled
# Keep bundled portraits in the renderer artifact directory on every start.
# This makes newly shipped looks available without rerunning the heavyweight
# installer or rebuilding TensorRT engines; AVTR-1 loads non-default looks on
# first selection.
AVTR1_FRAMES="$AVTR1_ROOT/artifacts/main/avatars_artifacts/reference_frames"
mkdir -p "$AVTR1_FRAMES"
for avatar_id in xiaoya_locket xiaoya xiaoya_idle xiaoya_beach_close xiaoya_beach sauna_portrait; do
  avatar_source="$ROOT/assets/looks/${avatar_id}.png"
  [[ -f "$avatar_source" ]] || die "missing avatar portrait $avatar_source"
  cp -f "$avatar_source" "$AVTR1_FRAMES/${avatar_id}.png"
  [[ -f "$ROOT/assets/looks/pasteback_mask_soft.png" ]] \
    && cp -f "$ROOT/assets/looks/pasteback_mask_soft.png" \
      "$AVTR1_FRAMES/${avatar_id}.pbmask.png"
done
# Expression portraits are separate render sources, not selectable characters.
# AVTR continues to drive their mouth from live audio; the gateway crossfades
# into them briefly for facial details its implicit keypoints cannot retarget.
expression_avatar_ids=()
for expression_source in "$ROOT"/assets/expressions/xiaoya_locket/reference-*.png; do
  [[ -f "$expression_source" ]] || continue
  expression_profile="$(basename "$expression_source" .png)"
  expression_profile="${expression_profile#reference-}"
  expression_profile="${expression_profile//-/_}"
  [[ "$expression_profile" == "neutral" ]] && continue
  expression_avatar_id="xiaoya_locket_expr_${expression_profile}"
  cp -f "$expression_source" "$AVTR1_FRAMES/${expression_avatar_id}.png"
  [[ -f "$ROOT/assets/looks/pasteback_mask_soft.png" ]] \
    && cp -f "$ROOT/assets/looks/pasteback_mask_soft.png" \
      "$AVTR1_FRAMES/${expression_avatar_id}.pbmask.png"
  expression_avatar_ids+=("$expression_avatar_id")
done
# Preload the expression sources so the first real conversation never stalls
# while extracting its portrait features. They are hidden from the avatar UI.
AVTR1_PRELOAD_IDS=("$AVTR1_AVATAR_ID" "${expression_avatar_ids[@]}")
export AVTR1_AVATAR_IDS="$(IFS=,; echo "${AVTR1_PRELOAD_IDS[*]}")"
if alive "$RUN/avtr1_renderer.pid"; then
  say "avtr1 renderer already running (pid $(cat "$RUN/avtr1_renderer.pid"))"
else
  say "start avtr1 renderer avatar=$AVTR1_AVATAR_ID ${AVTR1_OUT_W}x${AVTR1_OUT_H}"
  (
    cd "$AVTR1_ROOT"
    nohup setsid "$AVTR1_PY" -m uvicorn avtr1_renderer.api.app:app \
      --host 127.0.0.1 --port "${AVTR1_PORT:-18012}" \
      >"$LOG/avtr1_renderer.log" 2>&1 &
    echo $! >"$RUN/avtr1_renderer.pid"
  )
  sleep 0.8
  alive "$RUN/avtr1_renderer.pid" || die "avtr1 renderer exited immediately, see $LOG/avtr1_renderer.log"
fi
wait_port "${AVTR1_PORT:-18012}" "AVTR-1 renderer" 180
start_bg avatar_gw "$RUN/avatar_gw.pid" "$LOG/avatar_gw.log" \
  "$AVTR1_PY" "$ROOT/proxy/avtr1_gateway.py"
wait_port "${AVATAR_GW_PORT:-18011}" "avatar-gateway" 60

# 2) Low-latency WebRTC edge. Video is copied bit-for-bit from AVTR's H.264
# baseline stream. Only audio is encoded to Opus, so this adds no GPU load.
if [[ "$WEBRTC_ENABLED" != "0" ]]; then
  MEDIAMTX_BIN="$ROOT/third_party/mediamtx/mediamtx"
  [[ -x "$MEDIAMTX_BIN" ]] \
    || die "MediaMTX is missing; run ./scripts/install_mediamtx.sh"
  for value_name in WEBRTC_UDP_PORT WEBRTC_TCP_PORT MEDIAMTX_WHEP_PORT \
                    MEDIAMTX_RTSP_PORT MEDIAMTX_API_PORT MEDIAMTX_METRICS_PORT; do
    value="${!value_name}"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$value_name must be an integer"
  done
  [[ "$WEBRTC_PUBLIC_HOST" =~ ^[A-Za-z0-9._:-]+$ ]] \
    || die "WEBRTC_PUBLIC_HOST contains unsupported characters"
  MEDIAMTX_CONF="$RUN/mediamtx.yml"
  sed -e "s|__MEDIAMTX_WHEP_PORT__|$MEDIAMTX_WHEP_PORT|g" \
      -e "s|__MEDIAMTX_RTSP_PORT__|$MEDIAMTX_RTSP_PORT|g" \
      -e "s|__MEDIAMTX_API_PORT__|$MEDIAMTX_API_PORT|g" \
      -e "s|__MEDIAMTX_METRICS_PORT__|$MEDIAMTX_METRICS_PORT|g" \
      -e "s|__WEBRTC_UDP_PORT__|$WEBRTC_UDP_PORT|g" \
      -e "s|__WEBRTC_TCP_PORT__|$WEBRTC_TCP_PORT|g" \
      -e "s|__WEBRTC_PUBLIC_HOST__|$WEBRTC_PUBLIC_HOST|g" \
    "$ROOT/proxy/mediamtx.yml.tpl" > "$MEDIAMTX_CONF"
  start_bg mediamtx "$RUN/mediamtx.pid" "$LOG/mediamtx.log" \
    "$MEDIAMTX_BIN" "$MEDIAMTX_CONF"
  wait_port "$MEDIAMTX_WHEP_PORT" "MediaMTX WHEP" 30
  wait_port "$MEDIAMTX_RTSP_PORT" "MediaMTX RTSP" 30

  start_webrtc_publisher() {
    local name="$1" music="$2" path="$3"
    start_bg "$name" "$RUN/${name}.pid" "$LOG/${name}.log" \
      "$ROOT/scripts/webrtc_publisher.sh" "$music" "$path"
  }
  start_webrtc_publisher webrtc_music 1 avatar_music
  start_webrtc_publisher webrtc_voice 0 avatar_voice
  for _ in $(seq 1 40); do
    paths_json="$(curl -fsS --max-time 2 "http://127.0.0.1:${MEDIAMTX_API_PORT}/v3/paths/list" 2>/dev/null || true)"
    if grep -Eq '"name":"avatar_music"[^}]*"ready":true' <<<"$paths_json" \
        && grep -Eq '"name":"avatar_voice"[^}]*"ready":true' <<<"$paths_json"; then
      say "WebRTC H.264+Opus publishers ready"
      break
    fi
    sleep 0.25
  done
  grep -Eq '"name":"avatar_music"[^}]*"ready":true' <<<"${paths_json:-}" \
    && grep -Eq '"name":"avatar_voice"[^}]*"ready":true' <<<"${paths_json:-}" \
    || die "WebRTC publishers are not ready; see $LOG/webrtc_*.log"
fi

# 4) speech-to-speech
STT_BACKEND="${STT_BACKEND:-sensevoice}"
case "$STT_BACKEND" in
  sensevoice)
    export STT_BACKEND SENSEVOICE_LANGUAGE="${SENSEVOICE_LANGUAGE:-auto}" SENSEVOICE_USE_ITN="${SENSEVOICE_USE_ITN:-1}"
    export SENSEVOICE_EMOTION_ENABLED="${SENSEVOICE_EMOTION_ENABLED:-1}"
    STT_ARGS=(--stt paraformer
              --paraformer_stt_model_name "$SENSEVOICE_MODEL"
              --paraformer_stt_device cuda)
    ;;
  faster-whisper)
    STT_ARGS=(--stt faster-whisper
              --faster_whisper_stt_model_name "${FASTER_WHISPER_MODEL:-large-v3}"
              --faster_whisper_stt_gen_language zh)
    ;;
  *)
    die "unsupported STT_BACKEND=$STT_BACKEND (use sensevoice or faster-whisper)"
    ;;
esac
TTS_ARGS=(--tts qwen3
          --qwen3_tts_model_name "$TTS_MODEL"
          --qwen3_tts_language zh
          --qwen3_tts_streaming_chunk_size "${QWEN3_TTS_CHUNK_SIZE:-4}")
# These controls are consumed by the local emotion/prosody adapter rather than
# the upstream CLI argument parser, so they must be present in the child
# process environment. Values in config.env would otherwise appear configurable
# while silently falling back to Python defaults.
export TTS_EMOTION_ENABLED="${TTS_EMOTION_ENABLED:-1}"
export TTS_STYLE_INSTRUCT_ENABLED="${TTS_STYLE_INSTRUCT_ENABLED:-1}"
export TTS_PROSODY_ENABLED="${TTS_PROSODY_ENABLED:-1}"
export TTS_PROSODY_MAX_CLAUSE_CHARS="${TTS_PROSODY_MAX_CLAUSE_CHARS:-20}"
export TTS_TEMPERATURE="${TTS_TEMPERATURE:-0.65}"
export TTS_TOP_K="${TTS_TOP_K:-30}"
export TTS_TOP_P="${TTS_TOP_P:-0.85}"
export TTS_DO_SAMPLE="${TTS_DO_SAMPLE:-1}"
export TTS_REPETITION_PENALTY="${TTS_REPETITION_PENALTY:-1.05}"
if [[ -f "$REF_AUDIO" ]]; then
  TTS_ARGS+=(--qwen3_tts_ref_audio "$REF_AUDIO" --qwen3_tts_ref_text "$REF_TEXT")
else
  TTS_ARGS+=(--qwen3_tts_speaker Vivian)
fi

start_bg s2s "$RUN/s2s.pid" "$LOG/s2s.log" \
  env LD_LIBRARY_PATH="$S2S_LD_LIBRARY_PATH" \
  "$S2S_VENV/bin/python" "$ROOT/proxy/s2s_with_avatar_tee.py" \
    --device cuda \
    --mode realtime \
    --ws_host 127.0.0.1 \
    --ws_port "$S2S_PORT" \
    "${STT_ARGS[@]}" \
    --language zh \
    --llm_backend responses-api \
    --model_name "$LLM_NAME" \
    --responses_api_base_url "$LLM_BASE_URL" \
    --responses_api_api_key "$LLM_API_KEY" \
    --responses_api_stream \
    --responses_api_disable_thinking \
    --stream_batch_sentences "$LLM_STREAM_BATCH_SENTENCES" \
    --chat_size "$LLM_CHAT_SIZE" \
    --compact_history \
    --init_chat_prompt "$INIT_CHAT_PROMPT 若用户消息包含内部声学线索，把它只当作不确定的语气参考，自然提高共情程度；不要复述线索、不要解释情绪识别。" \
    "${TTS_ARGS[@]}" \
    --no_enable_live_transcription \
    --thresh "$VAD_THRESH" \
    --min_speech_ms "$MIN_SPEECH_MS" \
    --min_speech_continuation_ms "${MIN_SPEECH_CONTINUATION_MS:-128}" \
    --min_silence_ms "$MIN_SILENCE_MS" \
    --speech_pad_ms "$SPEECH_PAD_MS" \
    --speculative_reopen_ms "$REOPEN_MS" \
    --short_segment_merge_ms "$MERGE_MS"
wait_port "$S2S_PORT" "speech-to-speech" 300

# 5) frontend
start_bg web "$RUN/web.pid" "$LOG/web.log" \
  "$S2S_VENV/bin/uvicorn" server:app --app-dir "$FRONTEND" \
    --host 127.0.0.1 --port "$WEB_PORT" --no-access-log
wait_port "$WEB_PORT" "frontend" 30

# 6) nginx public proxy
NGINX_CONF="$RUN/nginx.conf"
sed -e "s|__ROOT__|$ROOT|g" \
    -e "s|__LISTEN_PORT__|${LISTEN_HTTP_PORT}|g" \
    -e "s|__NGINX_WORKERS__|${NGINX_WORKERS:-4}|g" \
    -e "s|__AVATAR_GW_PORT__|${AVATAR_GW_PORT:-18011}|g" \
    -e "s|__MEDIAMTX_WHEP_PORT__|${MEDIAMTX_WHEP_PORT:-18889}|g" \
    -e "s|__S2S_PORT__|${S2S_PORT}|g" \
    -e "s|__WEB_PORT__|${WEB_PORT}|g" \
  "$ROOT/proxy/nginx.conf.tpl" > "$NGINX_CONF"
if [[ -f "$RUN/nginx.pid" ]] && kill -0 "$(cat "$RUN/nginx.pid")" 2>/dev/null; then
  nginx -c "$NGINX_CONF" -s reload
  say "nginx reloaded"
else
  nginx -c "$NGINX_CONF"
  say "nginx started"
fi
wait_port "$LISTEN_HTTP_PORT" "nginx" 15

cat <<TXT

════════════════════════════════════════════════════════════
  Cyber Girlfriend is up

  Public URL:  https://${PUBLIC_IP}:${PUBLIC_HTTP_PORT}/

  First visit: accept the self-signed certificate warning, then allow the microphone.
  Mouth sync: ${AVATAR_BACKEND:-avtr1} (${AVTR1_AVATAR_ID:-$AVATAR_ID}). Click the page once to connect.
════════════════════════════════════════════════════════════
TXT
