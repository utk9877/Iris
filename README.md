# Iris

Local-first, on-device photo library — like Google Photos, but everything runs
on your own machine against an existing folder of images, non-destructively.

- **Backend** (`backend/`): Python 3.11 · FastAPI · SQLite (WAL) · ONNX Runtime
  (CoreML) · CLIP/SigLIP · InsightFace · PaddleOCR · OpenCV · hnswlib.
- **Frontend** (`frontend/`): Tauri (Rust) · React · TypeScript · TanStack
  Virtual · Tailwind.
- **Docs** (`docs/`): [architecture](docs/ARCHITECTURE.md) ·
  [phases](docs/PHASES.md).

See [`CLAUDE.md`](CLAUDE.md) for project constraints, structure, and conventions.

## Status

Scaffolding only — no application logic yet.

## Getting started

```bash
# Backend
cd backend && uv sync --all-extras

# Frontend (requires the Rust toolchain for Tauri)
cd frontend && npm install && npm run tauri dev
```
