"""Three-tier semantic search tests (synthetic embeddings, no model)."""

from __future__ import annotations

import sqlite3
import time

import numpy as np

from iris.config import Settings
from iris.db import apply_migrations, connect
from iris.db import photos as photos_db
from iris.embeddings.service import EmbeddingService
from iris.search import SearchResult, semantic_search


def _insert_photo(conn: sqlite3.Connection, name: str) -> int:
    now = time.time()
    cur = conn.execute(
        "INSERT INTO photos (path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "missing, created_at, updated_at) VALUES (?, ?, ?, '.jpg', 1, ?, ?, 0, ?, ?)",
        (f"/src/{name}", "/src", name, now, now, now, now),
    )
    assert cur.lastrowid is not None
    return cur.lastrowid


def _seed(settings: Settings) -> tuple[EmbeddingService, list[int]]:
    settings.ensure_dirs()
    conn = connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    apply_migrations(conn)
    ids = [_insert_photo(conn, f"p{i}.jpg") for i in range(3)]

    service = EmbeddingService(settings)
    vectors = np.eye(3, settings.embed_dim, dtype=np.float32)  # orthonormal, distinct
    start = service.store.append(vectors)
    conn.execute("BEGIN")
    for offset, pid in enumerate(ids):
        photos_db.set_embed(conn, pid, start + offset, time.time())
    conn.execute("COMMIT")
    conn.close()
    service.ensure_index()  # rebuild from DB + memmap
    return service, ids


def _query(
    service: EmbeddingService,
    settings: Settings,
    *,
    candidate_ids: set[int] | None,
    brute_max: int,
    filtered_max: int,
    k: int = 3,
) -> SearchResult:
    query = np.eye(1, settings.embed_dim, dtype=np.float32)[0]  # aligns with photo 0
    conn = connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    try:
        return semantic_search(
            service,
            conn,
            query,
            k=k,
            candidate_ids=candidate_ids,
            brute_max=brute_max,
            filtered_max=filtered_max,
        )
    finally:
        conn.close()


def test_unfiltered_uses_plain_hnsw(settings: Settings) -> None:
    service, ids = _seed(settings)
    result = _query(service, settings, candidate_ids=None, brute_max=2000, filtered_max=50000)
    assert result.tier == "hnsw"
    assert result.hits[0].photo_id == ids[0]


def test_small_filter_uses_brute_force(settings: Settings) -> None:
    service, ids = _seed(settings)
    result = _query(service, settings, candidate_ids=set(ids), brute_max=5, filtered_max=50)
    assert result.tier == "brute"
    assert result.hits[0].photo_id == ids[0]
    assert result.hits[0].score > 0.99


def test_medium_filter_uses_filtered_hnsw(settings: Settings) -> None:
    service, ids = _seed(settings)
    result = _query(service, settings, candidate_ids=set(ids), brute_max=2, filtered_max=50)
    assert result.tier == "filtered_hnsw"
    assert {h.photo_id for h in result.hits} <= set(ids)


def test_large_filter_uses_postfilter(settings: Settings) -> None:
    service, ids = _seed(settings)
    keep = {ids[0], ids[1]}
    result = _query(service, settings, candidate_ids=keep, brute_max=1, filtered_max=1)
    assert result.tier == "hnsw_postfilter"
    assert {h.photo_id for h in result.hits} <= keep
