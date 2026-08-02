# Packaging & distribution (Phase 6)

How to build, sign, and notarize a distributable Iris `.app`/`.dmg` for Apple Silicon.

> **Status.** The build *configuration* (PyInstaller spec, staging script, Tauri signing
> config, entitlements) is committed and reviewed. The actual signed/notarized build is
> **not** produced in CI or the dev container — it needs the **Rust toolchain** (not
> installed here) and an **Apple Developer certificate + notarization credentials** (which
> must never live in the repo). Run the steps below on a Mac that has both. See
> ARCHITECTURE §10 risk #3 for the underlying risks.

## Prerequisites

- **Rust** — `curl https://sh.rustup.rs -sSf | sh` (Tauri needs `cargo`/`rustc`).
- **Xcode command-line tools** — `xcode-select --install`.
- **Apple Developer Program** membership, and in your login keychain:
  - a *Developer ID Application* certificate (for signing outside the App Store), and
  - an app-specific password (or an App Store Connect API key) for `notarytool`.
- The Python build extras: `cd backend && uv sync --extra ml --extra packaging`.

## 1. Build the Python sidecar bundle

```bash
scripts/build_sidecar.sh
```

This runs PyInstaller against `backend/iris-sidecar.spec` (a `--onedir` bundle — see the
spec header for *why* onedir over onefile), **smoke-tests that the frozen binary prints the
`IRIS_SIDECAR_READY` handshake with no dev Python on PATH** (the Phase 6 done-when), and
copies `backend/dist/iris-sidecar/` to `frontend/src-tauri/binaries/iris-sidecar/`.

`tauri.conf.json` maps that folder into the app bundle via `bundle.resources`
(`binaries/iris-sidecar` → `iris-sidecar`), so the packaged shell finds the sidecar at
`Contents/Resources/iris-sidecar/iris-sidecar` and launches it exactly as in dev.

**Models are not bundled.** CLIP/SigLIP and the face models download on first run into the
app data dir (`~/Library/Application Support/Iris/models/`), keeping the installer small and
the model set updatable without a re-release. First launch therefore needs one-time network
access; everything after that is fully offline (ARCHITECTURE §3/§10).

## 2. Build, sign & notarize the app

Signing is driven by environment variables so no secret is written to disk or the repo:

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
export APPLE_ID="you@example.com"
export APPLE_PASSWORD="app-specific-password"   # or use APPLE_API_KEY / APPLE_API_ISSUER
export APPLE_TEAM_ID="TEAMID"

cd frontend
npm install
npm run tauri build          # signs with the identity above; hardened runtime + entitlements.plist
```

`bundle.macOS.entitlements` points at `src-tauri/entitlements.plist`, which disables
library validation and allows unsigned executable memory — required so the hardened runtime
lets the bundled onnxruntime/CoreML and insightface/opencv dylibs load. Tauri notarizes and
staples automatically when the `APPLE_*` credentials are present.

Artifacts land in `frontend/src-tauri/target/release/bundle/` (`.app` and `.dmg`).

## 3. Verify on a clean Mac

```bash
spctl -a -vvv --type install "path/to/Iris.dmg"     # -> "accepted / source=Notarized Developer ID"
codesign --verify --deep --strict --verbose=2 "Iris.app"
xcrun stapler validate "Iris.app"
```

The real acceptance test (PHASES Phase 6 done-when) is launching the `.dmg` on a Mac that
has **never had the dev toolchain**: the app opens, the sidecar handshake turns the health
dot green, and a scan runs.

## 4. Full-scale run & benchmarks (your library)

Packaging done, capture the headline scale numbers on the real library:

```bash
cd backend && uv run --extra embed --extra faces --extra ocr --extra geo \
    python ../scripts/bench_e2e.py --root "/path/to/Photos" --embed \
    --data-dir /tmp/iris-e2e-full
```

Records `e2e_ingest`, `peak_rss`, and `search_p95_full` to the `benchmarks` table (phase 6).
These must come from an actual run — never estimated (CLAUDE.md standing instruction #2).
```
