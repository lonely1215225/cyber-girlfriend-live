#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/config.env"

check() {
  local port="$1" name="$2"
  if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
    echo "  :$port/tcp  $name  OK"
  else
    echo "  :$port/tcp  $name  DOWN"
  fi
}

echo "Cyber Girlfriend status"
check "$LISTEN_HTTP_PORT" nginx
check "$WEB_PORT" frontend
check "$S2S_PORT" speech-to-speech
check "${AVTR1_PORT:-18012}" AVTR-1-renderer
check "${AVATAR_GW_PORT:-18011}" AVTR-1-gateway
echo
echo "Public URL: https://${PUBLIC_IP}:${PUBLIC_HTTP_PORT}/"
