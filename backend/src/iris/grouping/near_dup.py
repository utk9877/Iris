"""Near-duplicate detection: perceptual-hash proximity confirmed by CLIP (ARCHITECTURE §5).

Photos are sorted by their 64-bit ``phash``; each is compared against a small window of
neighbours in that order (similar hashes sort close together), and pairs within a Hamming
threshold are unioned. When both photos have a CLIP embedding, the pair must *also* clear a
cosine threshold — this rejects the occasional phash collision between unrelated images.
The windowed scan keeps the pass ~O(n·w) instead of O(n²); ``window`` is the recall knob.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

Row = dict[str, Any]
_MASK64 = 0xFFFFFFFFFFFFFFFF


def hamming64(a: int, b: int) -> int:
    """Bit-difference count between two hashes, treated as unsigned 64-bit."""
    return int((a ^ b) & _MASK64).bit_count()


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def find_near_dups(
    rows: Sequence[Row],
    *,
    hamming_max: int,
    cosine_min: float,
    vectors: dict[int, npt.NDArray[np.float32]] | None = None,
    window: int = 20,
) -> list[list[Row]]:
    """Group rows into near-duplicate components (each of size ≥ 2)."""
    candidates = [r for r in rows if r["phash"] is not None]
    if len(candidates) < 2:
        return []
    order = sorted(range(len(candidates)), key=lambda i: int(candidates[i]["phash"]) & _MASK64)
    uf = _UnionFind(len(candidates))
    for pos, i in enumerate(order):
        ri = candidates[i]
        for j in order[pos + 1 : pos + 1 + window]:
            rj = candidates[j]
            if hamming64(int(ri["phash"]), int(rj["phash"])) > hamming_max:
                continue
            if vectors is not None:
                vi = vectors.get(ri["embed_row"]) if ri["embed_row"] is not None else None
                vj = vectors.get(rj["embed_row"]) if rj["embed_row"] is not None else None
                if vi is not None and vj is not None and float(vi @ vj) < cosine_min:
                    continue  # phash-close but semantically different -> reject
            uf.union(i, j)

    components: dict[int, list[Row]] = {}
    for idx, row in enumerate(candidates):
        components.setdefault(uf.find(idx), []).append(row)
    return [group for group in components.values() if len(group) >= 2]
