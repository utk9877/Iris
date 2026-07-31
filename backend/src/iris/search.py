"""Three-tier semantic search (ARCHITECTURE §4).

Given a query vector and an optional metadata-filter candidate set, pick the cheapest
correct strategy:

* no filter                 -> plain HNSW over the whole index
* |candidates| <= T_small   -> exact brute force over the memmap
* |candidates| <= T_medium  -> filtered HNSW (allow-list)
* otherwise                 -> plain HNSW over-fetch + post-filter

Reciprocal-rank fusion is provided for combining with full-text (OCR) results; FTS
lands in Phase 4, so today only the semantic ranking flows through.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from iris.embeddings.clip import Vectors
from iris.embeddings.service import EmbeddingService

_SQL_VAR_LIMIT = 900  # stay under SQLite's default 999-variable cap


@dataclass
class SearchHit:
    photo_id: int
    score: float  # cosine similarity in [-1, 1]


@dataclass
class SearchResult:
    tier: str
    hits: list[SearchHit]


def _chunks(seq: Sequence[int], size: int = _SQL_VAR_LIMIT) -> Iterator[Sequence[int]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def build_candidates(
    conn: sqlite3.Connection,
    *,
    folder: str | None = None,
    date_from: float | None = None,
    date_to: float | None = None,
) -> set[int] | None:
    """Compile metadata filters to a candidate id set, or None if no filter is set."""
    clauses = ["missing = 0"]
    params: list[object] = []
    if folder:
        clauses.append("dir = ?")
        params.append(folder)
    if date_from is not None:
        clauses.append("sort_at >= ?")
        params.append(date_from)
    if date_to is not None:
        clauses.append("sort_at <= ?")
        params.append(date_to)
    if len(clauses) == 1:
        return None  # no real filter -> unfiltered search
    rows = conn.execute(f"SELECT id FROM photos WHERE {' AND '.join(clauses)}", params).fetchall()
    return {int(r[0]) for r in rows}


def _embed_rows_for(conn: sqlite3.Connection, ids: Sequence[int]) -> tuple[list[int], list[int]]:
    photo_ids: list[int] = []
    embed_rows: list[int] = []
    for chunk in _chunks(list(ids)):
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT id, embed_row FROM photos "
            f"WHERE embed_at IS NOT NULL AND embed_row IS NOT NULL AND id IN ({placeholders})",
            list(chunk),
        ).fetchall()
        for row in rows:
            photo_ids.append(int(row[0]))
            embed_rows.append(int(row[1]))
    return photo_ids, embed_rows


def _brute_force(
    service: EmbeddingService, conn: sqlite3.Connection, ids: set[int], query: Vectors, k: int
) -> SearchResult:
    photo_ids, embed_rows = _embed_rows_for(conn, list(ids))
    if not embed_rows:
        return SearchResult("brute", [])
    matrix = service.get_rows(embed_rows)
    sims = matrix @ query  # both L2-normalized -> cosine similarity
    order = np.argsort(-sims)[:k]
    return SearchResult("brute", [SearchHit(photo_ids[int(i)], float(sims[int(i)])) for i in order])


def semantic_search(
    service: EmbeddingService,
    conn: sqlite3.Connection,
    query_vec: Vectors,
    *,
    k: int,
    candidate_ids: set[int] | None,
    brute_max: int,
    filtered_max: int,
    ef: int = 128,
    overfetch: int = 5,
) -> SearchResult:
    query = np.asarray(query_vec, dtype=np.float32).reshape(-1)

    if candidate_ids is None:
        labels, dists = service.query(query, k, ef=ef)
        return SearchResult(
            "hnsw", [SearchHit(pid, 1.0 - d) for pid, d in zip(labels, dists, strict=True)]
        )
    if not candidate_ids:
        return SearchResult("empty", [])

    size = len(candidate_ids)
    if size <= brute_max:
        return _brute_force(service, conn, candidate_ids, query, k)
    if size <= filtered_max:
        labels, dists = service.query(query, k, ef=ef, allowed=candidate_ids)
        return SearchResult(
            "filtered_hnsw", [SearchHit(pid, 1.0 - d) for pid, d in zip(labels, dists, strict=True)]
        )
    labels, dists = service.query(query, k * overfetch, ef=max(ef, k * overfetch))
    hits = [
        SearchHit(pid, 1.0 - d)
        for pid, d in zip(labels, dists, strict=True)
        if pid in candidate_ids
    ][:k]
    return SearchResult("hnsw_postfilter", hits)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, k: int = 60
) -> list[tuple[int, float]]:
    """Fuse multiple ranked id lists (e.g. semantic + FTS) — used from Phase 4 on."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, photo_id in enumerate(ranking):
            scores[photo_id] = scores.get(photo_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: -item[1])
