#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/run"

if [[ -f "$ROOT/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/config.env"
fi

say() { echo "▸ $*"; }

pid_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

wait_dead() {
  local target="$1" tries="${2:-40}"
  local i
  for i in $(seq 1 "$tries"); do
    kill -0 -- "$target" 2>/dev/null || return 0
    sleep 0.25
  done
  return 1
}

terminate_pid() {
  local pid="$1"
  pid_alive "$pid" || return 0

  # Every background service started by start.sh owns a setsid process group.
  # Stop the whole group so workers cannot survive after their leader exits.
  local pgid
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
  if [[ "$pgid" == "$pid" ]]; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    if ! wait_dead "-$pgid"; then
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
  else
    kill -TERM "$pid" 2>/dev/null || true
    if ! wait_dead "$pid"; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
}

stop_pidfile() {
  local name="$1" pidfile="$2"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if pid_alive "$pid"; then
      say "stop $name ($pid)"
      terminate_pid "$pid"
    fi
    rm -f "$pidfile"
  fi
}

stop_matching() {
  local name="$1" pattern="$2"
  local pids pid
  pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
  [[ -n "$pids" ]] || return 0
  say "clean residual $name"
  while read -r pid; do
    [[ -n "$pid" && "$pid" != "$$" ]] && terminate_pid "$pid"
  done <<<"$pids"
}

if [[ -f "$RUN/nginx.pid" ]]; then
  nginx_pid="$(cat "$RUN/nginx.pid" 2>/dev/null || true)"
  nginx -c "$RUN/nginx.conf" -s quit 2>/dev/null || true
  nginx -c "$ROOT/proxy/nginx.conf" -s quit 2>/dev/null || true
  pid_alive "$nginx_pid" && wait_dead "$nginx_pid" 20 || true
  pid_alive "$nginx_pid" && terminate_pid "$nginx_pid"
  rm -f "$RUN/nginx.pid"
  say "stop nginx"
fi

stop_pidfile web "$RUN/web.pid"
stop_pidfile s2s "$RUN/s2s.pid"
stop_pidfile indextts "$RUN/indextts.pid"
stop_pidfile fish_s2 "$RUN/fish_s2.pid"
stop_pidfile webrtc_music "$RUN/webrtc_music.pid"
stop_pidfile webrtc_voice "$RUN/webrtc_voice.pid"
stop_pidfile mediamtx "$RUN/mediamtx.pid"
stop_pidfile avatar_gw "$RUN/avatar_gw.pid"
stop_pidfile avtr1_renderer "$RUN/avtr1_renderer.pid"
stop_pidfile thinkless "$RUN/thinkless.pid"
stop_pidfile grok_proxy "$RUN/grok_proxy.pid"
stop_pidfile log_guard "$RUN/log_guard.pid"

# PID files can be lost after a crash or manual cleanup. These root-scoped
# fallbacks make stop.sh idempotent without touching similarly named services
# belonging to another checkout.
stop_matching s2s "$ROOT/apps/speech/s2s_with_avatar_tee.py"
stop_matching s2s-legacy "$ROOT/proxy/s2s_with_avatar_tee.py"
stop_matching indextts "$ROOT/apps/speech/indextts25_server.py"
stop_matching fish-s2 "$ROOT/third_party/fish-speech/.venv/bin/python tools/api_server.py"
stop_matching frontend "uvicorn server:app --app-dir $ROOT/apps/web"
stop_matching frontend-legacy "uvicorn server:app --app-dir $ROOT/s2s/hf-realtime-voice"
stop_matching webrtc-publisher "ffmpeg.*127.0.0.1:.*livestream.flv.*rtsp://127.0.0.1:.*/avatar_"
stop_matching mediamtx "$ROOT/third_party/mediamtx/mediamtx $ROOT/run/mediamtx.yml"
stop_matching avtr1-renderer "$ROOT/third_party/avtr-1/.pixi/.*/bin/python -m uvicorn avtr1_renderer.api.app:app"
stop_matching avtr1-gateway "$ROOT/services/avatar/avtr1_gateway.py"
stop_matching avtr1-gateway-legacy "$ROOT/proxy/avtr1_gateway.py"
stop_matching thinkless "$ROOT/services/llm/ollama_thinkless.py"
stop_matching thinkless-legacy "$ROOT/proxy/ollama_thinkless.py"
stop_matching grok-proxy "/usr/local/bin/grok-reverse-proxy"
stop_matching log-guard "$ROOT/scripts/log_guard.sh $ROOT/logs"

# Ollama itself may be shared by other applications, so do not stop its daemon.
# Unload only the model configured for this project to release its GPU memory.
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
LLM_NAME="${LLM_NAME:-}"
if [[ -n "$LLM_NAME" && "$LLM_NAME" =~ ^[A-Za-z0-9._:/-]+$ ]] \
    && curl -fsS --max-time 3 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  say "unload Ollama model $LLM_NAME"
  unload_payload="$(printf '{"model":"%s","keep_alive":0}' "$LLM_NAME")"
  if curl -fsS --max-time 60 \
      -H 'Content-Type: application/json' \
      -d "$unload_payload" "$OLLAMA_URL/api/generate" >/dev/null; then
    for _ in $(seq 1 40); do
      loaded="$(curl -fsS --max-time 3 "$OLLAMA_URL/api/ps" 2>/dev/null || true)"
      grep -Fq "\"name\":\"$LLM_NAME\"" <<<"$loaded" || break
      sleep 0.25
    done
    say "Ollama model unloaded"
  else
    echo "! unable to unload Ollama model through $OLLAMA_URL" >&2
  fi
fi

LOCAL_LEAD_MODEL="${LOCAL_LEAD_MODEL:-}"
if [[ -n "$LOCAL_LEAD_MODEL" && "$LOCAL_LEAD_MODEL" != "$LLM_NAME" \
      && "$LOCAL_LEAD_MODEL" =~ ^[A-Za-z0-9._:/-]+$ ]] \
      && curl -fsS --max-time 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  say "unload Ollama model $LOCAL_LEAD_MODEL"
  unload_payload="$(printf '{"model":"%s","keep_alive":0}' "$LOCAL_LEAD_MODEL")"
  curl -fsS --max-time 20 -H 'Content-Type: application/json' \
    -d "$unload_payload" "$OLLAMA_URL/api/generate" >/dev/null || true
fi

say "Cyber Girlfriend stopped"
