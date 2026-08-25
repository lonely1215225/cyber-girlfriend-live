#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${MEDIAMTX_VERSION:-1.20.0}"
DEST="$ROOT/third_party/mediamtx"

if [[ -x "$DEST/mediamtx" ]] && [[ "$($DEST/mediamtx --version 2>/dev/null)" == "v$VERSION" ]]; then
  echo "▸ MediaMTX v${VERSION} already installed"
  exit 0
fi

case "$(uname -m)" in
  x86_64) ARCH=amd64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) echo "unsupported MediaMTX architecture: $(uname -m)" >&2; exit 1 ;;
esac

ARCHIVE="mediamtx_v${VERSION}_linux_${ARCH}.tar.gz"
BASE="https://github.com/bluenviron/mediamtx/releases/download/v${VERSION}"
TMP="$(mktemp -d)"
trap 'find "$TMP" -depth -delete 2>/dev/null || true' EXIT

echo "▸ download MediaMTX v${VERSION} (${ARCH})"
curl -fL --retry 3 --connect-timeout 15 "$BASE/$ARCHIVE" -o "$TMP/$ARCHIVE"
curl -fL --retry 3 --connect-timeout 15 "$BASE/checksums.sha256" -o "$TMP/checksums.sha256"
(
  cd "$TMP"
  grep -F "$ARCHIVE" checksums.sha256 | sha256sum --check -
)
tar -xzf "$TMP/$ARCHIVE" -C "$TMP" mediamtx LICENSE
mkdir -p "$DEST"
install -m 0755 "$TMP/mediamtx" "$DEST/mediamtx"
install -m 0644 "$TMP/LICENSE" "$DEST/LICENSE"
echo "▸ installed $($DEST/mediamtx --version)"
