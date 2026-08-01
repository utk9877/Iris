"""OCR ↔ semantic fusion in the /search endpoint (fake embedder, no CLIP model)."""

from __future__ import annotations

import sqlite3
import time
from typing import Any, cast

import numpy as np
from fastapi.testclient import TestClient

from iris.config import Settings
from iris.db import connect
from iris.db import ocr as ocr_db
from iris.db import photos as photos_db


class _FakeEmbedder:
    """Returns a fixed query vector aligned with photo 1's embedding."""

    def __init__(self, dim: int) -> None:
        self._vec = np.eye(1, dim, dtype=np.float32)[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec for _ in texts])


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _seed_photo(conn: sqlite3.Connection, pid: int) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO photos (id, path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "missing, created_at, updated_at) VALUES (?, ?, '/s', ?, '.jpg', 1, ?, ?, 0, ?, ?)",
        (pid, f"/s/p{pid}.jpg", f"p{pid}.jpg", now, now, now, now),
    )


def test_search_fuses_ocr_hits(client: TestClient, settings: Settings) -> None:
    service = cast(Any, client.app).state.embeddings
    service._embedder = _FakeEmbedder(settings.embed_dim)

    conn = _open(settings)
    _seed_photo(conn, 1)
    _seed_photo(conn, 2)
    vectors = np.eye(2, settings.embed_dim, dtype=np.float32)  # p1 aligns with the query
    start = service.store.append(vectors)
    conn.execute("BEGIN")
    photos_db.set_embed(conn, 1, start, time.time())
    photos_db.set_embed(conn, 2, start + 1, time.time())
    conn.execute("COMMIT")
    # Only photo 2 contains the searched word.
    ocr_db.set_ocr(conn, photo_id=2, text="quarterly invoice", regions=[], now=time.time())
    conn.close()
    service.ensure_index()

    resp = client.post("/search", json={"query": "invoice", "limit": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tier"] == "fused"  # OCR branch contributed
    ids = [item["id"] for item in body["items"]]
    assert set(ids) == {1, 2}
    # photo 2 (semantic rank 2 + OCR rank 1) should outrank photo 1 after fusion
    assert ids[0] == 2


def test_search_has_text_filter(client: TestClient, settings: Settings) -> None:
    service = cast(Any, client.app).state.embeddings
    service._embedder = _FakeEmbedder(settings.embed_dim)

    conn = _open(settings)
    _seed_photo(conn, 1)
    _seed_photo(conn, 2)
    vectors = np.eye(2, settings.embed_dim, dtype=np.float32)
    start = service.store.append(vectors)
    conn.execute("BEGIN")
    photos_db.set_embed(conn, 1, start, time.time())
    photos_db.set_embed(conn, 2, start + 1, time.time())
    conn.execute("COMMIT")
    ocr_db.set_ocr(conn, photo_id=2, text="some text", regions=[], now=time.time())
    conn.close()
    service.ensure_index()

    resp = client.post(
        "/search", json={"query": "anything", "limit": 10, "filters": {"has_text": True}}
    )
    ids = [item["id"] for item in resp.json()["items"]]
    assert ids == [2]  # only the photo with recognized text survives the filter
