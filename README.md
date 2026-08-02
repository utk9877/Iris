# Iris

Local-first, on-device photo library — like Google Photos, but everything runs on your
own machine against an existing folder of images, **non-destructively** (your originals are
never moved, renamed, or written to). No cloud, offline after first run.

## How it works

A **Tauri** desktop shell (React + TypeScript UI) launches a **FastAPI** Python sidecar
that does all the heavy lifting and talks to the UI over local HTTP.

- **Ingest pipeline** (per photo, crash-resumable via `*_at` stage markers):
  scan → hash + EXIF → decode + thumbnail + perceptual-hash → aesthetic/quality score →
  CLIP embedding → face detect/embed → OCR → grouping.
- **Storage:** SQLite (WAL) for metadata; float32 **memmaps** for CLIP/face vectors;
  **hnswlib** for vector search; content-addressed thumbnails/previews. All derived data
  lives in `~/Library/Application Support/Iris/` — never beside your originals.
- **Search:** natural-language text (CLIP) fused with OCR full-text (SQLite FTS5), plus
  date and offline place-name/GPS filters.
- **Organize:** people (face clustering), groups (events · bursts · near-duplicates ·
  themes), tags, and trip **triage** (aesthetic/quality/representativeness/subject scoring
  → deduped, diversified shortlists).

Models (CLIP, faces) download on first run into the app data dir; everything after is
offline. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/PHASES.md`](docs/PHASES.md) for detail, and [`CLAUDE.md`](CLAUDE.md) for conventions.

## Stack

- **Backend:** Python 3.11 · FastAPI · SQLite (WAL) · ONNX Runtime (CoreML) · CLIP/SigLIP ·
  InsightFace · Apple Vision OCR · OpenCV · hnswlib.
- **Frontend:** Tauri (Rust) · React · TypeScript · TanStack Virtual · Tailwind.

## Setup

```bash
cd backend  && uv sync --all-extras   # Python 3.11 via uv
cd frontend && npm install            # needs Node; Tauri also needs the Rust toolchain
```

## Run

**Desktop app** (needs Rust — `curl https://sh.rustup.rs -sSf | sh`):

```bash
cd frontend && npm run tauri dev      # launches the window + sidecar
```

**No-Rust dev fallback** (sidecar + UI in a browser), two terminals:

```bash
# 1) sidecar
cd backend && uv run --extra embed --extra faces --extra ocr --extra geo \
  iris-sidecar --host 127.0.0.1 --port 8756

# 2) UI  → then open http://localhost:1420
cd frontend && VITE_SIDECAR_URL=http://127.0.0.1:8756 npm run dev
```

Then add a library folder and scan from the UI. Packaging a signed `.dmg` is documented in
[`docs/PACKAGING.md`](docs/PACKAGING.md).
