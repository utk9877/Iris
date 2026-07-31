"""hnswlib vector index wrapper (ARCHITECTURE §4).

Cosine space over L2-normalized vectors; labels are ``photo_id`` so queries return
photo ids directly. The persisted index is a rebuildable cache — the source of truth
is the memmap + DB, so a missing/stale ``clip.hnsw`` is rebuilt from the store.
"""

from __future__ import annotations

from pathlib import Path

import hnswlib
import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]
Ids = npt.NDArray[np.int64]

_SLACK = 1024  # spare capacity so incremental adds don't resize every batch


class VectorIndex:
    def __init__(self, dim: int, *, ef_construction: int = 200, m: int = 16) -> None:
        self.dim = dim
        self._ef_construction = ef_construction
        self._m = m
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._ready = False

    def _ensure_capacity(self, extra: int) -> None:
        if not self._ready:
            self._index.init_index(
                max_elements=max(extra, _SLACK),
                ef_construction=self._ef_construction,
                M=self._m,
            )
            self._ready = True
            return
        needed = self._index.get_current_count() + extra
        if needed > self._index.get_max_elements():
            self._index.resize_index(needed + _SLACK)

    def add(self, vectors: Vectors, ids: Ids) -> None:
        if len(ids) == 0:
            return
        self._ensure_capacity(len(ids))
        self._index.add_items(vectors, ids)

    def build(self, vectors: Vectors, ids: Ids) -> None:
        """Rebuild the index from scratch (used to restore from the memmap)."""
        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        self._ready = False
        self.add(vectors, ids)

    def query(
        self, vector: Vectors, k: int, *, ef: int = 128, allowed: set[int] | None = None
    ) -> tuple[list[int], list[float]]:
        """Return (photo_ids, distances) for the k nearest neighbours.

        ``allowed`` restricts results to a candidate id set (tier-2 filtered HNSW).
        Cosine distance = 1 - cosine similarity.
        """
        size = self.size()
        if size == 0:
            return [], []
        k = min(k, size)
        self._index.set_ef(max(ef, k))
        label_filter = None if allowed is None else (lambda label: label in allowed)
        labels, distances = self._index.knn_query(vector, k=k, filter=label_filter)
        return labels[0].tolist(), distances[0].tolist()

    def size(self) -> int:
        return self._index.get_current_count() if self._ready else 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(path))

    def load(self, path: Path, *, count: int) -> None:
        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        self._index.load_index(str(path), max_elements=count + _SLACK)
        self._ready = True
