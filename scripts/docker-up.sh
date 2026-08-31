#!/usr/bin/env bash
# One-click Docker start for a GPU host with nvidia-container-toolkit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export COMPOSE_PROJECT_DIR="$ROOT"

die() { echo "✗ $*" >&2; exit 1; }

command -v docker >/dev/null || die "install Docker first"
docker compose version >/dev/null || die "install Docker Compose v2"
command -v nvidia-smi >/dev/null || die "need NVIDIA GPU and nvidia-smi"
if ! docker info 2>/dev/null | grep -Eqi 'nvidia|Runtimes:'; then
  echo "! warning: could not confirm an NVIDIA Docker runtime; compose will still request gpus: all"
fi

# Do not copy config.env.example here. install.sh writes PUBLIC_IP on first boot.
listen_port="19800"
if [[ -f "$ROOT/config.env" ]]; then
  listen_port="$(awk -F= '/^LISTEN_HTTP_PORT=/{print $2; exit}' "$ROOT/config.env" || true)"
  listen_port="${listen_port:-19800}"
fi
if ss -ltn "sport = :${listen_port}" 2>/dev/null | grep -q LISTEN; then
  die "port ${listen_port} is already in use. Stop the host stack first: $ROOT/scripts/stop.sh"
fi

docker compose -f "$ROOT/docker-compose.yml" up -d --build
echo "▸ container is up. Follow logs with: docker compose logs -f live"
