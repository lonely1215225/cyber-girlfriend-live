#!/usr/bin/env bash
# Build a deploy tarball (code + default looks, no models / venvs / logs).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${STAGE:-/tmp/cyber-girlfriend-pack}"
OUT="${OUT:-/root/cyber-girlfriend-deploy.tar.gz}"
rm -rf "$STAGE"
mkdir -p "$STAGE/cyber-girlfriend"

rsync -a \
  --exclude '.git/' \
  --exclude '/logs/' \
  --exclude '/run/' \
  --exclude '/models/' \
  --exclude '/.cache/' \
  --exclude 's2s/.venv/' \
  --exclude '__pycache__/' \
  --exclude 'third_party/avtr-1/.pixi/' \
  --exclude 'third_party/avtr-1/artifacts/' \
  --exclude 'third_party/avtr-1/.git/' \
  --exclude 'third_party/avtr-1/**/__pycache__/' \
  --exclude 'third_party/avtr-1/tests/' \
  --exclude 'proxy/certs/' \
  --exclude 'deploy/certs/' \
  --exclude 'proxy/nginx.conf' \
  --exclude 'scripts/make_release.sh' \
  --exclude 'config.env' \
  "$ROOT/" "$STAGE/cyber-girlfriend/"

PLUGIN_SO="$ROOT/third_party/avtr-1/artifacts/main/renderer_runtime_artifacts/libgrid_sample_3d_plugin.so"
if [[ -f "$PLUGIN_SO" ]]; then
  cp -f "$PLUGIN_SO" "$STAGE/cyber-girlfriend/assets/libgrid_sample_3d_plugin.so"
fi

mkdir -p "$STAGE/cyber-girlfriend/models" "$STAGE/cyber-girlfriend/logs" "$STAGE/cyber-girlfriend/run" \
  "$STAGE/cyber-girlfriend/deploy/certs"
chmod +x "$STAGE/cyber-girlfriend/install.sh" \
  "$STAGE/cyber-girlfriend/scripts/"*.sh

need=(
  apps/web/avatar-sync.js
  apps/web/style.css
  apps/web/worklets/mic-capture.js
  third_party/avtr-1/src/avtr1_renderer/models/avtr1.py
  third_party/avtr-1/src/avtr1_renderer/models/decoder.py
  third_party/avtr-1/src/avtr1_renderer/models/warp.py
  third_party/avtr-1/src/avtr1_renderer/components/alpha_to_luma_lut.npy
  third_party/avtr-1/scripts/build_engines.py
  scripts/log_guard.sh
  assets/idle.mp4
  assets/looks/xiaoya.png
  assets/looks/xiaoya_idle.png
  assets/looks/xiaoya_locket.png
  assets/looks/sauna_portrait.png
  assets/avatars/sauna_portrait.jpg
  assets/looks/pasteback_mask_soft.png
  deploy/nginx/nginx.conf.tpl
  proxy/avtr1_gateway.py
  assets/libgrid_sample_3d_plugin.so
)
for rel in "${need[@]}"; do
  [[ -f "$STAGE/cyber-girlfriend/$rel" ]] || { echo "pack missing $rel" >&2; exit 1; }
done

tar -C "$STAGE" -czf "$OUT" cyber-girlfriend
echo "wrote $OUT ($(du -h "$OUT" | awk '{print $1}'))"
