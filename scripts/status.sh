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

check_udp() {
  local port="$1" name="$2"
  if ss -lun "sport = :$port" 2>/dev/null | grep -q UNCONN; then
    echo "  :$port/udp  $name  OK"
  else
    echo "  :$port/udp  $name  DOWN"
  fi
}

echo "Cyber Girlfriend status"
check "$LISTEN_HTTP_PORT" nginx
check "$WEB_PORT" frontend
check "$S2S_PORT" speech-to-speech
check "${THINKLESS_PORT:-11435}" LLM-router
if [[ "${GROK_ENABLED:-0}" == "1" ]]; then
  check "${GROK_PROXY_PORT:-18080}" Grok-proxy
  if curl -fsS --max-time 2 "http://127.0.0.1:${GROK_PROXY_PORT:-18080}/healthz" >/dev/null; then
    echo "  provider   Grok OAuth session  OK"
  else
    echo "  provider   Grok OAuth session  DOWN"
  fi
fi
check "${AVTR1_PORT:-18012}" AVTR-1-renderer
check "${AVATAR_GW_PORT:-18011}" AVTR-1-gateway
if [[ "${WEBRTC_ENABLED:-1}" != "0" ]]; then
  check "${MEDIAMTX_WHEP_PORT:-18889}" MediaMTX-WHEP
  check "${MEDIAMTX_RTSP_PORT:-18554}" MediaMTX-RTSP
  check "${WEBRTC_TCP_PORT:-8190}" WebRTC-ICE
  check_udp "${WEBRTC_UDP_PORT:-8189}" WebRTC-ICE
  if curl -fsS --max-time 2 "http://127.0.0.1:${MEDIAMTX_API_PORT:-19997}/v3/paths/list" \
      | grep -q '"name":"avatar_music"'; then
    echo "  media      H264+Opus publishers  OK"
  else
    echo "  media      H264+Opus publishers  DOWN"
  fi
fi
echo
echo "Public URL: https://${PUBLIC_IP}:${PUBLIC_HTTP_PORT}/"
