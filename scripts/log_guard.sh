#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${1:?log directory required}"
MAX_BYTES="${LOG_MAX_BYTES:-2097152}"
INTERVAL="${LOG_CHECK_SECONDS:-60}"

while true; do
  for logfile in "$LOG_DIR"/*.log; do
    [[ -f "$logfile" ]] || continue
    [[ "$(basename "$logfile")" == "log_guard.log" ]] && continue
    size="$(stat -c %s "$logfile" 2>/dev/null || echo 0)"
    if (( size > MAX_BYTES )); then
      cp -f "$logfile" "$logfile.1.tmp"
      mv -f "$logfile.1.tmp" "$logfile.1"
      : >"$logfile"
      printf '%s rotated %s (%s bytes)\n' "$(date -Is)" "$(basename "$logfile")" "$size"
    fi
  done
  sleep "$INTERVAL"
done
