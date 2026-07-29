# CLAUDE.md

Guidance for Claude (and humans) working in this repository.

## Project summary

**Iris** is a local-first photo library app — like Google Photos, but everything
runs **on-device** against an existing folder of images, **non-destructively**
(files are never moved or copied out of place). The app indexes, searches, and
organizes photos entirely offline.

### Constraints

- **Scale:** ~100,000 photos, ~5 MB average (~500 GB of source data).
- **Hardware:** a single laptop — Apple Silicon Mac.
- **Non-destructive:** never move, rename, or copy the user's source images.
  All derived data (thumbnails, embeddings, index, DB) lives in app-managed
  locations, never alongside originals.
- **Local-first / offline:** no cloud dependency at runtime. All inference and
  search happen on-device.

### Stack

- **Backend:** Python 3.11, FastAPI, SQLite (WAL mode).
- **ML / CV:** ONNX Runtime (CoreML execution provider), CLIP / SigLIP,
  InsightFace (SCRFD + ArcFace), LAION aesthetic predictor, PaddleOCR, OpenCV.
- **Vector search:** hnswlib.
- **Frontend:** React + TypeScript + TanStack Virtual + Tailwind.
- **Desktop shell:** Tauri (Rust).
- **Packaging:** PyInstaller for the Python sidecar.

## Repository structure

```
.
├── backend/            # Python FastAPI sidecar (the "brain": indexing, ML, search)
│   ├── src/iris/       #   application package (type-hinted Python)
│   ├── tests/          #   pytest suite
│   ├── pyproject.toml  #   uv-managed project + ruff/pytest/mypy config
│   └── .python-version #   pinned to 3.11
├── frontend/           # Tauri + React + TypeScript desktop app (the UI shell)
│   ├── src/            #   React/TS source
│   └── src-tauri/      #   Rust (Tauri) shell; hosts the Python sidecar
├── scripts/            # dev/ops helper scripts (setup, build, packaging)
├── docs/               # architecture & planning docs
│   ├── ARCHITECTURE.md #   system design (filled in during planning)
│   └── PHASES.md       #   phased delivery plan (filled in during planning)
└── CLAUDE.md           # this file
```

The Tauri Rust shell (`frontend/src-tauri/`) launches the FastAPI Python sidecar,
which does the heavy lifting (file scanning, ONNX inference, SQLite, hnswlib).
The React frontend talks to the sidecar over local HTTP.

## Setup

- **Backend:** `cd backend && uv sync --all-extras` (Python 3.11 via uv).
- **Frontend:** `cd frontend && npm install`. Requires the Rust toolchain
  (`rustup`) installed to build/run Tauri — not yet installed in this
  environment; run `npm run tauri dev` once Rust is present.

> Dependency versions in `pyproject.toml` are intentionally **unpinned** during
> scaffolding. Run `uv lock` / `uv sync` once the environment is validated on
> Apple Silicon (note: `paddlepaddle` / `insightface` may need attention there).

## Coding conventions

- **Python type hints are required** on all functions (enforced via ruff `ANN`
  and `mypy --strict`). No untyped public functions.
- **Linting/formatting:** `ruff` (`uv run ruff check .` and `ruff format`).
- **Tests:** `pytest` (`uv run pytest`). Write tests alongside features.
- **Frontend:** TypeScript strict mode; React function components + hooks.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, …).

## Standing instructions

1. **Stay on the approved plan.** If execution diverges from an approved plan
   mid-task, **stop and re-enter Plan Mode** rather than improvising.

2. **Never fabricate benchmark numbers.** All performance claims must come from
   an **actual run against real data**, logged to the `jobs/benchmarks` table.
   Do not estimate, extrapolate, or invent throughput/latency figures.

3. **Non-destructive, always.** Never move, rename, delete, or write into the
   user's source image folders. Derived artifacts go only in app-managed
   directories.
