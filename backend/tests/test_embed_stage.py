"""Embed stage integration: orchestrator embeds thumbnails via an injected fake CLIP."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from iris.config import Settings
from iris.db import apply_migrations, connect
from iris.db import library as library_db
from iris.embeddings.clip import CLIPEmbedder
from iris.embeddings.service import EmbeddingService
from iris.ingest.orchestrator import IngestManager


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeVision:
    """Returns a 512-d vector whose first 3 dims are the per-channel means.

    Distinct solid colours -> distinct vector directions, so the index can tell them
    apart (no real weights / network needed).
    """

    def get_inputs(self) -> list[Any]:
        return [_FakeInput("pixel_values")]

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        batch = next(iter(input_feed.values()))
        out = np.zeros((batch.shape[0], 512), dtype=np.float32)
        out[:, :3] = batch.mean(axis=(2, 3))
        out[:, 3] = 0.1  # avoid an all-zero vector
        return [out]


class _FakeText:
    def get_inputs(self) -> list[Any]:
        return [_FakeInput("input_ids")]

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        n = next(iter(input_feed.values())).shape[0]
        return [np.ones((n, 512), dtype=np.float32)]


class _FakeTokenizer:
    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": np.zeros((len(texts), 77), dtype=np.int64)}


def _service(settings: Settings) -> EmbeddingService:
    embedder = CLIPEmbedder(_FakeVision(), _FakeText(), _FakeTokenizer(), dim=512)
    return EmbeddingService(settings, embedder=embedder)


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _wait(manager: IngestManager, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not manager.is_running():
            return
        time.sleep(0.1)
    raise TimeoutError


def test_embed_stage_populates_store_and_index(settings: Settings, tmp_path: Path) -> None:
    settings.ensure_dirs()
    settings.decode_workers = 2
    conn = _open(settings)
    apply_migrations(conn)
    src = tmp_path / "src"
    src.mkdir()
    colors = [(230, 20, 20), (20, 230, 20), (20, 20, 230)]
    for i, color in enumerate(colors):
        Image.new("RGB", (128, 96), color).save(src / f"c{i}.png")
    library_db.add_root(conn, str(src), time.time())
    conn.close()

    service = _service(settings)
    manager = IngestManager(settings, service)
    manager.start()
    _wait(manager)

    conn = _open(settings)
    try:
        assert manager.status()["job"]["state"] == "done"
        embedded = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE embed_at IS NOT NULL AND embed_row IS NOT NULL"
        ).fetchone()[0]
        assert embedded == 3
        rowmap = dict(
            conn.execute("SELECT id, embed_row FROM photos WHERE embed_row IS NOT NULL").fetchall()
        )
    finally:
        conn.close()

    # memmap holds 3 vectors; the persisted index file exists.
    assert service.store.count == 3
    assert (settings.index_dir / "clip.hnsw").exists()

    # Querying the index with a photo's own stored vector returns that photo.
    for photo_id, embed_row in rowmap.items():
        vec = service.store.get_rows([embed_row])[0]
        labels, distances = service.query(vec, k=1)
        assert labels[0] == photo_id
        assert distances[0] < 1e-4

    # Delete the index file: a fresh service rebuilds it from the memmap + DB.
    (settings.index_dir / "clip.hnsw").unlink()
    rebuilt = _service(settings)
    pid, row = next(iter(rowmap.items()))
    labels, _ = rebuilt.query(rebuilt.store.get_rows([row])[0], k=1)
    assert labels[0] == pid
