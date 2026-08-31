#!/usr/bin/env bash
# Keep the container in the foreground after the usual one-click start.
set -euo pipefail

ROOT="${ROOT:-/opt/cyber-girlfriend}"
cd "$ROOT"

say() { echo "▸ $*"; }
die() { echo "✗ $*" >&2; exit 1; }

command -v nvidia-smi >/dev/null || die "this image needs an NVIDIA GPU and nvidia-container-toolkit"
nvidia-smi >/dev/null || die "GPU is not visible inside the container"

ENGINE="$ROOT/third_party/avtr-1/artifacts/main/speech2motion_runtime_artifacts_cc/avtr1_decode_fp16.engine"
MEDIAMTX="$ROOT/third_party/mediamtx/mediamtx"
if [[ ! -x "$ROOT/s2s/.venv/bin/speech-to-speech" || ! -f "$ENGINE" || ! -x "$MEDIAMTX" ]]; then
  say "first-run install; models and TensorRT engines can take a long time"
  "$ROOT/install.sh"
fi

# install.sh writes PUBLIC_IP into a new config.env. Only fall back to the
# example after install, so a first boot does not pin PUBLIC_IP=127.0.0.1.
if [[ ! -f "$ROOT/config.env" ]]; then
  say "config.env still missing; copying example (set PUBLIC_IP before going public)"
  cp "$ROOT/config.env.example" "$ROOT/config.env"
fi

"$ROOT/scripts/start.sh"

stop_stack() {
  "$ROOT/scripts/stop.sh" || true
  exit 0
}
trap stop_stack TERM INT

alive_pidfile() {
  local pidfile="$1"
  local pid
  [[ -f "$pidfile" ]] || return 1
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

# start.sh daemonizes nginx and the Python workers. Stay alive only while the
# public edge and the two processes that actually talk are still up.
while true; do
  if alive_pidfile "$ROOT/run/nginx.pid" \
      && alive_pidfile "$ROOT/run/web.pid" \
      && alive_pidfile "$ROOT/run/s2s.pid"; then
    sleep 2
    continue
  fi
  die "nginx, web, or speech-to-speech exited; see $ROOT/logs"
done
