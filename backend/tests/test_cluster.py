"""Chinese Whispers face clustering tests (synthetic embeddings, known groups)."""

from __future__ import annotations

import numpy as np

from iris.faces.cluster import chinese_whispers, cluster_embeddings


def _grouped_unit_vectors(groups: int, per_group: int, dim: int, seed: int = 0) -> np.ndarray:
    """Tight clusters: each group is a random center plus small noise, L2-normalized."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((groups, dim)).astype(np.float32)
    out = []
    for g in range(groups):
        noise = 0.03 * rng.standard_normal((per_group, dim)).astype(np.float32)
        out.append(centers[g] + noise)
    v = np.concatenate(out, axis=0)
    return np.asarray(v / np.linalg.norm(v, axis=1, keepdims=True), dtype=np.float32)


def test_chinese_whispers_two_components() -> None:
    # Two disconnected triangles -> two labels.
    adjacency = {
        1: [(2, 1.0), (3, 1.0)],
        2: [(1, 1.0), (3, 1.0)],
        3: [(1, 1.0), (2, 1.0)],
        4: [(5, 1.0), (6, 1.0)],
        5: [(4, 1.0), (6, 1.0)],
        6: [(4, 1.0), (5, 1.0)],
    }
    labels = chinese_whispers(adjacency, seed=1)
    assert labels[1] == labels[2] == labels[3]
    assert labels[4] == labels[5] == labels[6]
    assert labels[1] != labels[4]


def test_cluster_embeddings_recovers_groups() -> None:
    groups, per_group, dim = 3, 15, 64
    matrix = _grouped_unit_vectors(groups, per_group, dim, seed=7)
    ids = list(range(100, 100 + groups * per_group))
    assignments = cluster_embeddings(ids, matrix, k=10, edge_threshold=0.5)

    # Exactly 3 clusters, and each true group maps to a single cluster.
    assert len(set(assignments.values())) == 3
    for g in range(groups):
        block = ids[g * per_group : (g + 1) * per_group]
        assert len({assignments[i] for i in block}) == 1
    # Cluster indices are contiguous starting at 0.
    assert set(assignments.values()) == {0, 1, 2}


def test_cluster_embeddings_empty() -> None:
    assert cluster_embeddings([], np.empty((0, 8), dtype=np.float32)) == {}


def test_edge_threshold_controls_pose_merging() -> None:
    """Same person, two 'poses' at cosine ~0.4: strict threshold splits, loose merges."""
    dim = 64
    rng = np.random.default_rng(5)
    a = np.zeros(dim, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(dim, dtype=np.float32)
    b[0], b[1] = 0.4, float(np.sqrt(1 - 0.16))  # cos(a, b) == 0.4
    pose_a = a + 0.02 * rng.standard_normal((8, dim)).astype(np.float32)
    pose_b = b + 0.02 * rng.standard_normal((8, dim)).astype(np.float32)
    v = np.concatenate([pose_a, pose_b])
    matrix = np.asarray(v / np.linalg.norm(v, axis=1, keepdims=True), dtype=np.float32)
    ids = list(range(16))

    strict = cluster_embeddings(ids, matrix, k=10, edge_threshold=0.5)  # old default
    loose = cluster_embeddings(ids, matrix, k=10, edge_threshold=0.35)  # new default
    assert len(set(strict.values())) == 2  # angle splits the person in two
    assert len(set(loose.values())) == 1  # ...but the recalibrated threshold reunites them
