#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/config.env"

music="${1:?music flag required}"
path="${2:?MediaMTX path required}"
[[ "$music" =~ ^[01]$ ]] || { echo "invalid music flag" >&2; exit 2; }
[[ "$path" =~ ^avatar_(music|voice)$ ]] || { echo "invalid MediaMTX path" >&2; exit 2; }

child=""
stop_child() {
  [[ "$child" =~ ^[0-9]+$ ]] || return 0
  kill -TERM "$child" 2>/dev/null || true
  wait "$child" 2>/dev/null || true
}
trap 'stop_child; exit 0' TERM INT

while true; do
  ffmpeg -hide_banner -loglevel warning \
    -fflags +genpts+nobuffer -flags low_delay \
    -analyzeduration 1000000 -probesize 1000000 \
    -i "http://127.0.0.1:${AVATAR_GW_PORT:-18011}/livestream.flv?music=$music" \
    -map 0:v:0 -c:v copy \
    -map 0:a:0 -af "aresample=async=1000:first_pts=0,asetpts=N/SR/TB" \
    -c:a libopus -ar 48000 -ac 1 \
    -b:a "${WEBRTC_OPUS_BITRATE:-48000}" -application lowdelay -frame_duration 20 \
    -fec 1 -packet_loss "${WEBRTC_PACKET_LOSS_PERCENT:-5}" \
    -flush_packets 1 -muxdelay 0 -rtsp_transport tcp -f rtsp \
    "rtsp://127.0.0.1:${MEDIAMTX_RTSP_PORT:-18554}/$path" &
  child=$!
  started_at=$SECONDS
  failures=0
  while kill -0 "$child" 2>/dev/null; do
    sleep 5
    (( SECONDS - started_at < 15 )) && continue
    if curl -fsS --max-time 2 \
      "http://127.0.0.1:${MEDIAMTX_API_PORT:-19997}/v3/paths/get/$path" \
      | grep -q '"ready":true'; then
      failures=0
    else
      failures=$((failures + 1))
      if (( failures >= 2 )); then
        echo "publisher path $path is not ready; restarting ffmpeg" >&2
        stop_child
        break
      fi
    fi
  done
  wait "$child" 2>/dev/null || true
  child=""
  sleep 1
done
