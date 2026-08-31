#!/usr/bin/env bash
# Keep the container in the foreground after the usual one-click start.
set -euo pipefail

ROOT="${ROOT:-/opt/cyber-girlfriend}"
cd "$ROOT"

say() { echo "▸ $*"; }
die() { echo "✗ $*" >&2; exit 1; }

command -v nvidia-smi >/dev/null || die "this image needs an NVIDIA GPU and nvidia-container-toolkit"
nvidia-smi >/dev/null || die "GPU is not visible inside the container"

if [[ ! -f "$ROOT/config.env" ]]; then
  say "creating config.env from the example"
  cp "$ROOT/config.env.example" "$ROOT/config.env"
fi

ENGINE="$ROOT/third_party/avtr-1/artifacts/main/speech2motion_runtime_artifacts_cc/avtr1_decode_fp16.engine"
if [[ ! -x "$ROOT/s2s/.venv/bin/speech-to-speech" || ! -f "$ENGINE" ]]; then
  say "first-run install; models and TensorRT engines can take a long time"
  "$ROOT/install.sh"
fi

"$ROOT/scripts/start.sh"

stop_stack() {
  "$ROOT/scripts/stop.sh" || true
  exit 0
}
trap stop_stack TERM INT

# start.sh daemonizes nginx and the Python workers. Stay alive with them.
while true; do
  nginx_pid=""
  [[ -f "$ROOT/run/nginx.pid" ]] && nginx_pid="$(cat "$ROOT/run/nginx.pid" 2>/dev/null || true)"
  if [[ "$nginx_pid" =~ ^[0-9]+$ ]] && kill -0 "$nginx_pid" 2>/dev/null; then
    sleep 2
    continue
  fi
  die "nginx exited; see $ROOT/logs"
done
