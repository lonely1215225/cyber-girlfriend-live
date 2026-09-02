#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$ROOT/config.env" ]]; then
  echo "✗ missing $ROOT/config.env" >&2
  exit 1
fi
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
TTS_BACKEND="${TTS_BACKEND:-indextts25}"
if [[ "$TTS_BACKEND" == "indextts" || "$TTS_BACKEND" == "indextts25" ]]; then
  tts_code="$(
    curl -fsS --max-time 3 "${INDEXTTS_URL:-http://127.0.0.1:18782}/healthz" 2>/dev/null || true
  )"
  if grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' <<<"$tts_code"; then
    echo "  tts        IndexTTS-2.5  OK"
  else
    echo "  tts        IndexTTS-2.5  DOWN"
  fi
elif [[ "$TTS_BACKEND" == "fish_s2" ]]; then
  check "${FISH_S2_PORT:-18781}" Fish-S2
else
  echo "  tts        $TTS_BACKEND  UNKNOWN"
fi
if [[ "${WEBRTC_ENABLED:-1}" != "0" ]]; then
  check "${MEDIAMTX_WHEP_PORT:-18889}" MediaMTX-WHEP
  check "${MEDIAMTX_RTSP_PORT:-18554}" MediaMTX-RTSP
  check "${WEBRTC_TCP_PORT:-8190}" WebRTC-ICE
  check_udp "${WEBRTC_UDP_PORT:-8189}" WebRTC-ICE
  paths_json="$(curl -fsS --max-time 2 "http://127.0.0.1:${MEDIAMTX_API_PORT:-19997}/v3/paths/list" 2>/dev/null || true)"
  if grep -Eq '"name":"avatar_music"[^}]*"ready":true' <<<"$paths_json" \
      && grep -Eq '"name":"avatar_voice"[^}]*"ready":true' <<<"$paths_json"; then
    echo "  media      H264+Opus publishers  OK"
  else
    echo "  media      H264+Opus publishers  DOWN"
  fi
fi
echo
echo "Public URL: https://${PUBLIC_IP}:${PUBLIC_HTTP_PORT}/"
