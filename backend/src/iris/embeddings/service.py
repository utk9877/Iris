"""Embedding service: owns the vector store, hnswlib index, and CLIP embedder.

Shared between the ingest embed stage (writer) and the search API (reader). The
index is a rebuildable cache: on first use it loads ``clip.hnsw`` if it matches the
DB, otherwise it rebuilds from the memmap (satisfying "delete clip.hnsw and restart
rebuilds from memmap"). A lock serializes index mutation vs. queries.
"""

from __future__ import annotations

import logging
import sqlite3
import threading

import numpy as np

from iris.config import Settings
from iris.db import connect
from iris.embeddings.clip import CLIPEmbedder, Vectors
from iris.embeddings.index import Ids, VectorIndex
from iris.embeddings.store import VectorStore

logger = logging.getLogger("iris.embeddings")


class EmbeddingService:
    def __init__(self, settings: Settings, *, embedder: CLIPEmbedder | None = None) -> None:
        self._settings = settings
        self._store = VectorStore(
            settings.embeddings_dir / "clip.f32", settings.embed_dim, settings.embed_model
        )
        self._index = VectorIndex(settings.embed_dim)
        self._index_path = settings.index_dir / "clip.hnsw"
        self._embedder = embedder
        self._lock = threading.Lock()
        self._index_ready = False

    @property
    def store(self) -> VectorStore:
        return self._store

    def embedder(self) -> CLIPEmbedder:
        """Lazily load the CLIP model (may download on first use)."""
        if self._embedder is None:
            logger.info(
                "loading CLIP model %s (first use may download)", self._settings.embed_model
            )
            self._embedder = CLIPEmbedder.load(
                self._settings.models_dir, self._settings.embed_model, self._settings.embed_dim
            )
        return self._embedder

    # ------------------------------------------------------------------ index
    def _embedded_pairs(self) -> list[tuple[int, int]]:
        conn = connect(
            self._settings.db_path, busy_timeout_ms=self._settings.sqlite_busy_timeout_ms
        )
        try:
            rows = conn.execute(
                "SELECT embed_row, id FROM photos "
                "WHERE embed_at IS NOT NULL AND embed_row IS NOT NULL ORDER BY embed_row"
            ).fetchall()
        finally:
            conn.close()
        return [(int(r[0]), int(r[1])) for r in rows]

    def _load_or_rebuild(self) -> None:
        pairs = self._embedded_pairs()
        if self._index_path.exists():
            try:
                self._index.load(self._index_path, count=len(pairs))
                if self._index.size() == len(pairs):
                    self._index_ready = True
                    return
                logger.warning("index size %d != db %d; rebuilding", self._index.size(), len(pairs))
            except Exception:
                logger.warning("failed to load index; rebuilding from memmap", exc_info=True)
        rows: Ids = np.array([p[1] for p in pairs], dtype=np.int64)
        vectors: Vectors = (
            self._store.get_rows([p[0] for p in pairs])
            if pairs
            else np.empty((0, self._settings.embed_dim), dtype=np.float32)
        )
        self._index.build(vectors, rows)
        self._index_ready = True

    def ensure_index(self) -> None:
        if self._index_ready:
            return
        with self._lock:
            if not self._index_ready:
                self._load_or_rebuild()

    def add_to_index(self, vectors: Vectors, ids: Ids) -> None:
        self.ensure_index()
        with self._lock:
            self._index.add(vectors, ids)

    def query(
        self, vector: Vectors, k: int, *, ef: int = 128, allowed: set[int] | None = None
    ) -> tuple[list[int], list[float]]:
        self.ensure_index()
        with self._lock:
            return self._index.query(vector, k, ef=ef, allowed=allowed)

    def persist_index(self) -> None:
        with self._lock:
            if self._index_ready:
                self._index.save(self._index_path)

    def matrix(self) -> Vectors:
        return self._store.matrix()

    def get_rows(self, rows: list[int]) -> Vectors:
        return self._store.get_rows(rows)


def open_readonly_conn(settings: Settings) -> sqlite3.Connection:
    """Convenience for callers needing a short-lived DB connection."""
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
