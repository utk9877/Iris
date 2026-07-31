"""Append-only float32 vector store backed by a memmap file (ARCHITECTURE §1/§3).

Vectors live in ``clip.f32`` (flat ``count x dim`` float32) with a ``clip.meta.json``
sidecar recording dim/dtype/count/model. Kept out of SQLite so brute-force search can
mmap a contiguous matrix and hnswlib can consume it directly. Row index == the value
stored in ``photos.embed_row``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]

_ITEMSIZE = 4  # float32


class VectorStore:
    """Single-writer, many-reader memmap vector store."""

    def __init__(self, data_path: Path, dim: int, model_id: str) -> None:
        self.data_path = data_path
        self.meta_path = data_path.parent / f"{data_path.stem}.meta.json"
        self.dim = dim
        self.model_id = model_id
        self._count = 0
        self._load_or_init()

    def _load_or_init(self) -> None:
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
            if meta.get("dim") != self.dim or meta.get("model_id") != self.model_id:
                raise ValueError(
                    "embedding store mismatch: stored "
                    f"{meta.get('model_id')}/{meta.get('dim')} vs {self.model_id}/{self.dim}. "
                    "Delete the embeddings/ dir to re-embed under the new model."
                )
            self._count = int(meta["count"])
        else:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            self.data_path.touch(exist_ok=True)
            self._count = 0
            self._write_meta()

    def _write_meta(self) -> None:
        self.meta_path.write_text(
            json.dumps(
                {
                    "dim": self.dim,
                    "dtype": "float32",
                    "model_id": self.model_id,
                    "count": self._count,
                }
            )
        )

    @property
    def count(self) -> int:
        return self._count

    def append(self, vectors: Vectors) -> int:
        """Append ``k`` vectors; return the starting row index they occupy.

        The data file is written and fsynced before the meta count is bumped, so a
        crash leaves at most orphaned (never mis-pointed) rows — the caller commits
        ``embed_row`` to SQLite only after this returns.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (k, {self.dim}) vectors, got {vectors.shape}")
        payload = np.ascontiguousarray(vectors, dtype=np.float32)
        start = self._count
        with self.data_path.open("r+b") as handle:
            handle.seek(start * self.dim * _ITEMSIZE)
            handle.write(payload.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        self._count += int(payload.shape[0])
        self._write_meta()
        return start

    def matrix(self) -> Vectors:
        """Read-only ``(count, dim)`` memmap view of all vectors."""
        if self._count == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.memmap(self.data_path, dtype=np.float32, mode="r", shape=(self._count, self.dim))

    def get_rows(self, rows: Sequence[int]) -> Vectors:
        """Gather specific rows into a contiguous array (for brute-force search)."""
        if not rows:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.asarray(self.matrix()[list(rows)], dtype=np.float32)
