#!/usr/bin/env bash
# Build the PyInstaller sidecar bundle and stage it for the Tauri app (Phase 6).
#
# Produces backend/dist/iris-sidecar/ (onedir) and copies it under
# frontend/src-tauri/binaries/ so `tauri build` bundles it as a resource. See
# docs/PACKAGING.md for the full signing + notarization runbook.
#
# Usage:  scripts/build_sidecar.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$ROOT/backend"
DEST="$ROOT/frontend/src-tauri/binaries"

echo "==> Building sidecar with PyInstaller (this bundles the ML stack; takes a few minutes)"
cd "$BACKEND"
uv sync --extra ml --extra packaging
uv run pyinstaller iris-sidecar.spec --noconfirm --clean

echo "==> Smoke-testing the frozen binary starts with no dev Python"
# --port 0 binds an ephemeral port; we just want the readiness handshake then exit.
timeout 30 ./dist/iris-sidecar/iris-sidecar --port 0 --data-dir "$(mktemp -d)" \
  | grep -m1 "IRIS_SIDECAR_READY" && echo "    handshake OK"

echo "==> Staging bundle into $DEST"
rm -rf "$DEST/iris-sidecar"
mkdir -p "$DEST"
cp -R "$BACKEND/dist/iris-sidecar" "$DEST/iris-sidecar"

echo "==> Done. Next: cd frontend && npm run tauri build  (see docs/PACKAGING.md for signing)"
