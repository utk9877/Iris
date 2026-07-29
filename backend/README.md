# Iris backend

Local-first, on-device photo library backend. FastAPI sidecar driving ONNX
Runtime (CoreML) inference over an existing folder of images, indexed in SQLite
(WAL) and searched with hnswlib.

## Development

```bash
uv sync --all-extras   # create venv + install deps (see note in pyproject.toml)
uv run ruff check .
uv run pytest
```

The Python version is pinned to 3.11 via `.python-version` / uv.
