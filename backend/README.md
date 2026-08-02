# Iris backend

The FastAPI **sidecar** — the "brain" of [Iris](../README.md). Launched by the Tauri shell
(or standalone in dev), it runs the ingest pipeline and ML inference and serves the UI over
local HTTP. Nothing here ever writes to your source images.

- **Pipeline:** scan → hash + EXIF → decode + thumbnail + phash → aesthetic/quality score →
  CLIP embedding → face detect/embed → OCR → grouping. Every stage records a per-photo
  `*_at` marker, so a crash/cancel just resumes the unfinished photos.
- **Storage:** SQLite (WAL) for metadata; float32 memmaps for CLIP/face vectors; hnswlib for
  vector search; content-addressed thumbnails/previews. All under the app data dir.
- **API:** search (semantic + OCR + date/place), photos, people, groups, tags, triage,
  and maintenance (storage stats, memmap compaction). Browse `/docs` when it's running.

Optional-dependency **extras** keep installs lean and CI fast: `embed` (CLIP + hnswlib),
`faces` (InsightFace + OpenCV), `ocr` (Apple Vision, macOS-only), `geo` (offline places).
Missing extras degrade gracefully (that stage is skipped).

## Develop

```bash
uv sync --all-extras            # venv + all deps (Python pinned to 3.11 via uv)
uv run ruff check . && uv run ruff format --check .
uv run mypy                     # strict
uv run pytest                   # test suite
```

CI parity: features must pass with only `uv sync --extra embed` installed (the extra-gated
stages skip cleanly), so run that before pushing if you touch those paths.

## Run the sidecar

```bash
uv run --extra embed --extra faces --extra ocr --extra geo \
  iris-sidecar --host 127.0.0.1 --port 8756
```

It prints `IRIS_SIDECAR_READY {…}` once bound. Overridable via `IRIS_*` env vars
(e.g. `IRIS_DATA_DIR`, `IRIS_PORT`). Benchmarks live in [`../scripts/`](../scripts);
config/layout details are in [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).
