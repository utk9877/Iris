"""Memmap vector store + hnswlib index tests (synthetic vectors, no model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iris.embeddings.index import VectorIndex
from iris.embeddings.store import VectorStore


def _unit(rows: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((rows, dim)).astype(np.float32)
    return np.asarray(v / np.linalg.norm(v, axis=1, keepdims=True), dtype=np.float32)


def test_store_append_and_read(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "clip.f32", dim=8, model_id="m1")
    a = _unit(3, 8, seed=1)
    b = _unit(2, 8, seed=2)
    assert store.append(a) == 0
    assert store.append(b) == 3
    assert store.count == 5
    np.testing.assert_allclose(store.get_rows([0, 1, 2]), a, atol=1e-6)
    np.testing.assert_allclose(store.get_rows([3, 4]), b, atol=1e-6)
    assert store.matrix().shape == (5, 8)


def test_store_persists_count_across_reopen(tmp_path: Path) -> None:
    VectorStore(tmp_path / "clip.f32", dim=4, model_id="m1").append(_unit(6, 4))
    reopened = VectorStore(tmp_path / "clip.f32", dim=4, model_id="m1")
    assert reopened.count == 6


def test_store_model_mismatch_raises(tmp_path: Path) -> None:
    VectorStore(tmp_path / "clip.f32", dim=4, model_id="m1")
    with pytest.raises(ValueError, match="mismatch"):
        VectorStore(tmp_path / "clip.f32", dim=4, model_id="OTHER")


def test_index_nearest_is_self(tmp_path: Path) -> None:
    vecs = _unit(50, 16, seed=7)
    ids = np.arange(100, 150, dtype=np.int64)  # photo ids offset from row index
    index = VectorIndex(dim=16)
    index.build(vecs, ids)
    assert index.size() == 50
    labels, distances = index.query(vecs[10], k=3)
    assert labels[0] == 110  # ids[10] == 100 + 10
    assert distances[0] < 1e-4


def test_index_filtered_query(tmp_path: Path) -> None:
    vecs = _unit(40, 16, seed=3)
    ids = np.arange(40, dtype=np.int64)
    index = VectorIndex(dim=16)
    index.build(vecs, ids)
    allowed = {5, 6, 7}
    labels, _ = index.query(vecs[5], k=3, allowed=allowed)
    assert set(labels) <= allowed


def test_index_save_load_roundtrip(tmp_path: Path) -> None:
    vecs = _unit(30, 12, seed=11)
    ids = np.arange(30, dtype=np.int64)
    index = VectorIndex(dim=12)
    index.build(vecs, ids)
    path = tmp_path / "clip.hnsw"
    index.save(path)

    restored = VectorIndex(dim=12)
    restored.load(path, count=30)
    assert restored.size() == 30
    labels, _ = restored.query(vecs[20], k=1)
    assert labels[0] == 20
