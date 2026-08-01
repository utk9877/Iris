"""Semantic (theme) grouping over CLIP embeddings (ARCHITECTURE §5).

The originally-planned HDBSCAN is replaced by the same kNN-graph + Chinese Whispers
community detector already used for faces (``iris.faces.cluster``): it needs no cluster
count up front, adds no new heavy dependency (numpy + hnswlib only), and keeps CI lean.
Themes cross-cut the temporal hierarchy; tiny groups are dropped as noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from iris.faces.cluster import cluster_embeddings

Vectors = npt.NDArray[np.float32]


@dataclass
class Theme:
    members: list[int]  # photo ids, representative first
    rep_id: int
    cohesion: float  # mean cosine of members to the theme centroid


def cluster_semantic(
    ids: Sequence[int],
    matrix: Vectors,
    *,
    k: int,
    edge_threshold: float,
    min_size: int,
) -> list[Theme]:
    """Cluster embeddings into themes of at least ``min_size`` photos."""
    if len(ids) < min_size:
        return []
    labels = cluster_embeddings(list(ids), matrix, k=k, edge_threshold=edge_threshold)
    row_of = {int(pid): i for i, pid in enumerate(ids)}

    grouped: dict[int, list[int]] = {}
    for photo_id, label in labels.items():
        grouped.setdefault(label, []).append(int(photo_id))

    themes: list[Theme] = []
    for members in grouped.values():
        if len(members) < min_size:
            continue
        block = matrix[[row_of[m] for m in members]]
        centroid = block.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        sims = block @ centroid
        order = np.argsort(-sims)
        ranked = [members[int(i)] for i in order]
        themes.append(Theme(ranked, ranked[0], float(sims.mean())))
    themes.sort(key=lambda t: len(t.members), reverse=True)
    return themes
