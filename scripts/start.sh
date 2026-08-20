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
SENSEVOICE_MODEL="${SENSEVOICE_MODEL:-models/sensevoice/SenseVoiceSmall}"
if [[ "$SENSEVOICE_MODEL" == models/* ]]; then
  SENSEVOICE_MODEL="$ROOT/$SENSEVOICE_MODEL"
fi

say() { echo "▸ $*"; }
die() { echo "✗ $*" >&2; exit 1; }

alive() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] || return 1
  kill -0 "$(cat "$pidfile")" 2>/dev/null
}

start_bg() {
  local name="$1" pidfile="$2" logfile="$3"
  shift 3
  if alive "$pidfile"; then
    say "$name already running (pid $(cat "$pidfile"))"
    return 0
  fi
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
export MENTION_RESEARCH_TIMEOUT="${MENTION_RESEARCH_TIMEOUT:-20}"
export MENTION_PRICE_CACHE_SECONDS="${MENTION_PRICE_CACHE_SECONDS:-30}"
export MENTION_NEWS_CACHE_SECONDS="${MENTION_NEWS_CACHE_SECONDS:-180}"
export NEWS_RSS_ENABLED="${NEWS_RSS_ENABLED:-1}"
export NEWS_GOOGLE_RSS_ENABLED="${NEWS_GOOGLE_RSS_ENABLED:-1}"
export NEWS_RSS_TIMEOUT="${NEWS_RSS_TIMEOUT:-8}"
export NEWS_RSS_CACHE_SECONDS="${NEWS_RSS_CACHE_SECONDS:-120}"
export NEWS_RSS_MAX_ITEMS="${NEWS_RSS_MAX_ITEMS:-10}"
export NEWS_RSS_MAX_AGE_HOURS="${NEWS_RSS_MAX_AGE_HOURS:-168}"
export NEWS_RSS_FEEDS="${NEWS_RSS_FEEDS:-}"
export NEWS_RSS_ALLOWED_HOSTS="${NEWS_RSS_ALLOWED_HOSTS:-}"
export S2S_INTERNAL_WS_URL="ws://127.0.0.1:${S2S_PORT}/v1/realtime"
export TTS_EMOTION_ENABLED="${TTS_EMOTION_ENABLED:-1}"
export TTS_STYLE_INSTRUCT_ENABLED="${TTS_STYLE_INSTRUCT_ENABLED:-1}"
export TTS_TEMPERATURE="${TTS_TEMPERATURE:-0.75}"
export TTS_TOP_K="${TTS_TOP_K:-40}"
export TTS_TOP_P="${TTS_TOP_P:-0.90}"
export TTS_DO_SAMPLE="${TTS_DO_SAMPLE:-1}"
export TTS_REPETITION_PENALTY="${TTS_REPETITION_PENALTY:-1.05}"
export AVATAR_TEE_PREROLL_MS="${AVATAR_TEE_PREROLL_MS:-400}"
export STARTUP_GREETING="${STARTUP_GREETING:-}"
export IDLE_PROMPT="${IDLE_PROMPT:-}"
export IDLE_PROMPT_MIN_SECONDS="${IDLE_PROMPT_MIN_SECONDS:-35}"
export IDLE_PROMPT_MAX_SECONDS="${IDLE_PROMPT_MAX_SECONDS:-55}"
export PROACTIVE_NEWS_MIN_SECONDS="${PROACTIVE_NEWS_MIN_SECONDS:-90}"
export PROACTIVE_NEWS_MAX_SECONDS="${PROACTIVE_NEWS_MAX_SECONDS:-150}"
export OLLAMA_URL LLM_NAME THINKLESS_PORT LLM_NUM_CTX LLM_NUM_PREDICT LLM_KEEP_ALIVE LLM_PREWARM
export LLM_CHAT_SIZE="${LLM_CHAT_SIZE:-12}"
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
export AVTR1_AVATAR_IDS="${AVTR1_AVATAR_ID:-xiaoya}"
export AVTR1_AVATAR_ID="${AVTR1_AVATAR_ID:-xiaoya}"
export AVTR1_BG_ID="${AVTR1_BG_ID:-plain_white}"
export AVTR1_OUT_H="${AVTR1_OUT_H:-1280}"
export AVTR1_OUT_W="${AVTR1_OUT_W:-720}"
export AVTR1_H264_BITRATE="${AVTR1_H264_BITRATE:-1200000}"
export AVTR1_CFG_SELF_AUDIO="${AVTR1_CFG_SELF_AUDIO:-2.3}"
export AVTR1_CFG_OTHER_AUDIO="${AVTR1_CFG_OTHER_AUDIO:-2.0}"
export AVTR1_CFG_KP="${AVTR1_CFG_KP:-3.0}"
export AVTR1_NOISE_ALPHA="${AVTR1_NOISE_ALPHA:-1.5}"
export AVTR1_NOISE_TRUNC_Z="${AVTR1_NOISE_TRUNC_Z:-1.0}"
export AVTR1_IDLE_NOISE_ALPHA="${AVTR1_IDLE_NOISE_ALPHA:-10.0}"
export AVTR1_IDLE_NOISE_TRUNC_Z="${AVTR1_IDLE_NOISE_TRUNC_Z:-0.25}"
export AVTR1_MOTION_AUDIO_RMS="${AVTR1_MOTION_AUDIO_RMS:-80}"
export AVTR1_MOTION_LISTEN_RMS="${AVTR1_MOTION_LISTEN_RMS:-450}"
export AVTR1_MOTION_ACTIVE_HOLD_SECONDS="${AVTR1_MOTION_ACTIVE_HOLD_SECONDS:-0.8}"
export AVTR1_URL="http://127.0.0.1:${AVTR1_PORT:-18012}"
export AVTR1_LOCAL_TEE_URL="http://127.0.0.1:${AVATAR_GW_PORT:-18011}"
export LOAD_BALANCER_URL=disabled
# Keep bundled portraits in the renderer artifact directory on every start.
# This makes newly shipped looks available without rerunning the heavyweight
# installer or rebuilding TensorRT engines; AVTR-1 loads non-default looks on
# first selection.
AVTR1_FRAMES="$AVTR1_ROOT/artifacts/main/avatars_artifacts/reference_frames"
mkdir -p "$AVTR1_FRAMES"
for avatar_id in xiaoya xiaoya_idle xiaoya_beach_close xiaoya_beach xiaoya_locket; do
  avatar_source="$ROOT/assets/looks/${avatar_id}.png"
  [[ -f "$avatar_source" ]] || die "missing avatar portrait $avatar_source"
  cp -f "$avatar_source" "$AVTR1_FRAMES/${avatar_id}.png"
  [[ -f "$ROOT/assets/looks/pasteback_mask_soft.png" ]] \
    && cp -f "$ROOT/assets/looks/pasteback_mask_soft.png" \
      "$AVTR1_FRAMES/${avatar_id}.pbmask.png"
done
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
