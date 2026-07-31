"""Face clustering via Chinese Whispers over a kNN similarity graph (ARCHITECTURE §6).

Chinese Whispers is a near-linear, label-propagation community detector that does not
need the number of clusters up front — a good fit for "how many people are in this
library?". We build the kNN graph with the same hnswlib index used for search.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from iris.embeddings.index import VectorIndex

Vectors = npt.NDArray[np.float32]
Adjacency = dict[int, list[tuple[int, float]]]

DEFAULT_K = 20
DEFAULT_EDGE_THRESHOLD = 0.5  # cosine; ARCHITECTURE §6


def chinese_whispers(
    adjacency: Mapping[int, Sequence[tuple[int, float]]],
    *,
    iterations: int = 20,
    seed: int = 0,
) -> dict[int, int]:
    """Propagate labels until stable; returns node -> raw label (a node id)."""
    labels: dict[int, int] = {node: node for node in adjacency}
    nodes = list(adjacency)
    rng = random.Random(seed)
    for _ in range(iterations):
        rng.shuffle(nodes)
        changed = False
        for node in nodes:
            weights: dict[int, float] = {}
            for neighbor, weight in adjacency[node]:
                weights[labels[neighbor]] = weights.get(labels[neighbor], 0.0) + weight
            if not weights:
                continue
            best = max(weights, key=lambda label: weights[label])
            if labels[node] != best:
                labels[node] = best
                changed = True
        if not changed:
            break
    return labels


def _knn_adjacency(
    ids: Sequence[int], matrix: Vectors, *, k: int, edge_threshold: float
) -> Adjacency:
    index = VectorIndex(matrix.shape[1])
    index.build(matrix, np.asarray(ids, dtype=np.int64))
    adjacency: Adjacency = {int(i): [] for i in ids}
    neighbours = min(k + 1, len(ids))  # +1 because the nearest hit is the node itself
    for row, node in enumerate(ids):
        labels, dists = index.query(matrix[row], k=neighbours, ef=max(64, neighbours))
        for neighbor, dist in zip(labels, dists, strict=True):
            if neighbor == node:
                continue
            similarity = 1.0 - dist
            if similarity >= edge_threshold:
                # undirected: record the edge on both endpoints
                adjacency[int(node)].append((int(neighbor), similarity))
                adjacency[int(neighbor)].append((int(node), similarity))
    return adjacency


def cluster_embeddings(
    ids: Sequence[int],
    matrix: Vectors,
    *,
    k: int = DEFAULT_K,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    iterations: int = 20,
    seed: int = 0,
) -> dict[int, int]:
    """Cluster face embeddings; returns ``face_id -> contiguous cluster index``.

    Cluster indices are assigned largest-cluster-first for stable, readable ids.
    """
    if len(ids) == 0:
        return {}
    adjacency = _knn_adjacency(ids, matrix, k=k, edge_threshold=edge_threshold)
    raw = chinese_whispers(adjacency, iterations=iterations, seed=seed)

    members: dict[int, list[int]] = {}
    for node, label in raw.items():
        members.setdefault(label, []).append(node)
    ordered = sorted(members.values(), key=len, reverse=True)
    return {node: cluster_idx for cluster_idx, group in enumerate(ordered) for node in group}
