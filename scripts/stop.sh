#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/run"

stop_pidfile() {
  local name="$1" pidfile="$2"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "stop $name ($pid)"
      kill "$pid" 2>/dev/null || true
      sleep 0.4
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
}

if [[ -f "$RUN/nginx.pid" ]]; then
  nginx -c "$RUN/nginx.conf" -s stop 2>/dev/null || true
  nginx -c "$ROOT/proxy/nginx.conf" -s stop 2>/dev/null || true
  rm -f "$RUN/nginx.pid"
  echo "stop nginx"
fi

stop_pidfile web "$RUN/web.pid"
stop_pidfile s2s "$RUN/s2s.pid"
stop_pidfile thinkless "$RUN/thinkless.pid"
stop_pidfile log_guard "$RUN/log_guard.pid"
stop_pidfile avatar_gw "$RUN/avatar_gw.pid"
stop_pidfile avtr1_renderer "$RUN/avtr1_renderer.pid"

pkill -f "proxy/s2s_with_avatar_tee.py" 2>/dev/null || true
pkill -f "avtr1_renderer.api.app" 2>/dev/null || true
pkill -f "proxy/avtr1_gateway.py" 2>/dev/null || true

echo "stopped"
