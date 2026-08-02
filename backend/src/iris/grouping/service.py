"""Grouping service: compute all four group layers and persist them (ARCHITECTURE §5).

Runs after embeddings so near-dup confirmation and semantic themes can use CLIP vectors.
Each layer is rebuilt wholesale (``db.groups.replace_groups``), so calling ``rebuild``
after new ingest simply refreshes the groupings. Missing embeddings degrade gracefully:
near-dup falls back to phash-only and semantic themes are skipped.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from iris.config import Settings
from iris.db import groups as groups_db
from iris.db import photos as photos_db
from iris.db.groups import GroupSpec
from iris.embeddings.service import EmbeddingService
from iris.grouping.events import detect_bursts, segment_events
from iris.grouping.near_dup import find_near_dups
from iris.grouping.semantic import cluster_semantic

logger = logging.getLogger("iris.grouping")

Row = dict[str, Any]


def _resolution(row: Row) -> int:
    return int((row["width"] or 0) * (row["height"] or 0))


def _rank_key(row: Row) -> tuple[float, float, int, int]:
    """Sort key for picking the best frame of a group (highest first).

    Prefers the Phase 5 technical ``quality`` then ``aesthetic`` scores; falls back to
    resolution (then lowest id) when scores are absent (NULL -> 0.0), so groups built
    before scoring still get a sensible representative.
    """
    return (
        float(row.get("quality") or 0.0),
        float(row.get("aesthetic") or 0.0),
        _resolution(row),
        -int(row["id"]),
    )


def _rep_best(rows: list[Row]) -> int:
    """Representative = the highest-quality frame of the group (ARCHITECTURE §5/§7)."""
    return int(max(rows, key=_rank_key)["id"])


def _day_key(ts: float | None) -> str | None:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts, tz=dt.UTC).date().isoformat()


class GroupingService:
    def __init__(self, settings: Settings, embeddings: EmbeddingService | None = None) -> None:
        self._settings = settings
        self._embeddings = embeddings

    def _vectors_by_row(self, rows: list[Row]) -> dict[int, npt.NDArray[np.float32]] | None:
        """Map ``embed_row -> unit vector`` for rows that have one (or None if no service)."""
        if self._embeddings is None:
            return None
        embed_rows = sorted({r["embed_row"] for r in rows if r["embed_row"] is not None})
        if not embed_rows:
            return {}
        matrix = self._embeddings.get_rows(embed_rows)
        return {er: matrix[i] for i, er in enumerate(embed_rows)}

    # ------------------------------------------------------------------ layers
    def _event_specs(self, rows: list[Row]) -> tuple[list[GroupSpec], list[list[Row]]]:
        events = segment_events(
            rows,
            gap_seconds=self._settings.event_gap_seconds,
            gps_km=self._settings.event_gps_km,
        )
        specs: list[GroupSpec] = []
        for event in events:
            if len(event) < 2:
                continue
            specs.append(
                GroupSpec(
                    kind="event",
                    key=_day_key(event[0]["sort_at"]),
                    rep_photo_id=_rep_best(event),
                    start_at=event[0]["sort_at"],
                    end_at=event[-1]["sort_at"],
                    members=[(r["id"], float(i)) for i, r in enumerate(event)],
                )
            )
        return specs, events

    def _burst_specs(self, events: list[list[Row]]) -> list[GroupSpec]:
        specs: list[GroupSpec] = []
        for event in events:
            for burst in detect_bursts(event, gap_seconds=self._settings.burst_gap_seconds):
                specs.append(
                    GroupSpec(
                        kind="burst",
                        rep_photo_id=_rep_best(burst),
                        start_at=burst[0]["sort_at"],
                        end_at=burst[-1]["sort_at"],
                        members=[(r["id"], float(i)) for i, r in enumerate(burst)],
                    )
                )
        return specs

    def _near_dup_specs(
        self, rows: list[Row], vectors: dict[int, npt.NDArray[np.float32]] | None
    ) -> list[GroupSpec]:
        components = find_near_dups(
            rows,
            hamming_max=self._settings.near_dup_hamming,
            cosine_min=self._settings.near_dup_cosine,
            vectors=vectors,
        )
        specs: list[GroupSpec] = []
        for comp in components:
            ranked = sorted(comp, key=_rank_key, reverse=True)  # best (highest quality) first
            specs.append(
                GroupSpec(
                    kind="near_dup",
                    rep_photo_id=ranked[0]["id"],
                    score=float(len(ranked)),
                    members=[(r["id"], float(i)) for i, r in enumerate(ranked)],
                )
            )
        return specs

    def _semantic_specs(self, rows: list[Row]) -> list[GroupSpec]:
        if self._embeddings is None:
            return []
        embedded = [r for r in rows if r["embed_row"] is not None]
        if len(embedded) < self._settings.semantic_min_size:
            return []
        ids = [r["id"] for r in embedded]
        matrix = self._embeddings.get_rows([r["embed_row"] for r in embedded])
        themes = cluster_semantic(
            ids,
            matrix,
            k=self._settings.semantic_k,
            edge_threshold=self._settings.semantic_edge_threshold,
            min_size=self._settings.semantic_min_size,
        )
        return [
            GroupSpec(
                kind="semantic",
                key=f"theme-{idx}",
                rep_photo_id=theme.rep_id,
                score=theme.cohesion,
                members=[(pid, float(rank)) for rank, pid in enumerate(theme.members)],
            )
            for idx, theme in enumerate(themes)
        ]

    # ------------------------------------------------------------------ driver
    def rebuild(self, conn: Any) -> dict[str, int]:
        """Recompute every group layer from current photos. Returns per-kind counts."""
        rows = photos_db.fetch_grouping_rows(conn)
        vectors = self._vectors_by_row(rows)

        event_specs, events = self._event_specs(rows)
        counts = {
            "event": groups_db.replace_groups(conn, "event", event_specs),
            "burst": groups_db.replace_groups(conn, "burst", self._burst_specs(events)),
            "near_dup": groups_db.replace_groups(
                conn, "near_dup", self._near_dup_specs(rows, vectors)
            ),
            "semantic": groups_db.replace_groups(conn, "semantic", self._semantic_specs(rows)),
        }
        logger.info("grouping rebuilt: %s", counts)
        return counts
