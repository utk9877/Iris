"""Trip-triage service: four scores → percentile-normalize → weight → diversify (§7).

Given a *scope* (an event/burst/theme group, or a date range) this computes, per photo,
four raw scores:

* **quality**  — technical score from ingest (:mod:`iris.scoring`), read off ``photos``.
* **aesthetic** — aesthetic score from ingest, read off ``photos``.
* **representativeness** — cosine of the photo's CLIP vector to the scope centroid.
* **subject** — face presence × face quality, bonused for named (known) people.

Each is **percentile-rank normalized within the scope** (outlier-robust, makes the four
comparable), then combined by the preset weights. For picking presets the shortlist is
first **collapsed** so near-duplicate / burst clusters contribute only their best frame
(ARCHITECTURE §5/§7 ordering), then diversified with **MMR** over CLIP embeddings. The
``delete-candidates`` preset inverts this: it ranks the *redundant and weak* frames
(all-but-best of each cluster, plus low-quality singletons) and skips MMR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from iris.config import Settings
from iris.db import faces as faces_db
from iris.db import groups as groups_db
from iris.db import photos as photos_db
from iris.embeddings.service import EmbeddingService
from iris.triage.presets import TriagePreset

logger = logging.getLogger("iris.triage")

_F = npt.NDArray[np.float64]
_HIGH = 0.66  # normalized-score threshold for "this is a strength" in reason text
_NAMED_BONUS = 0.6  # extra subject weight for a face belonging to a named person
_REDUNDANCY_BONUS = 0.5  # added to delete-candidate badness for a non-best cluster frame


@dataclass(frozen=True)
class TriageItem:
    photo_id: int
    score: float  # combined pick-score (keep presets) or badness (delete-candidates)
    quality: float  # all four below are percentile-normalized in [0, 1]
    aesthetic: float
    representativeness: float
    subject: float
    reason: str
    redundant: bool  # a non-best member of a near-dup / burst cluster


@dataclass(frozen=True)
class TriageResult:
    preset: str
    scope_size: int  # photos in scope
    considered: int  # candidates after collapse (keep) / redundancy filter (delete)
    items: list[TriageItem]


def _percentile_rank(values: _F) -> _F:
    """Average-rank percentile of each value, scaled to ``[0, 1]`` (neutral 0.5 if n<2)."""
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.full(1, 0.5, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks / (n - 1)


class TriageService:
    def __init__(self, settings: Settings, embeddings: EmbeddingService) -> None:
        self._settings = settings
        self._embeddings = embeddings

    # ------------------------------------------------------------------ vectors
    def _vectors(self, rows: dict[int, dict[str, Any]]) -> dict[int, _F]:
        """Map ``photo_id -> unit CLIP vector`` for photos that have an embedding."""
        embed_rows = sorted({r["embed_row"] for r in rows.values() if r["embed_row"] is not None})
        if not embed_rows:
            return {}
        matrix = np.asarray(self._embeddings.get_rows(embed_rows), dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-12, None)
        by_row = {er: matrix[i] for i, er in enumerate(embed_rows)}
        return {
            pid: by_row[r["embed_row"]] for pid, r in rows.items() if r["embed_row"] is not None
        }

    # --------------------------------------------------------------- raw scores
    def _representativeness(self, ids: list[int], vectors: dict[int, _F]) -> _F:
        """Cosine of each photo to the scope centroid; median-filled where no vector."""
        raw = np.full(len(ids), np.nan, dtype=np.float64)
        present = [i for i, pid in enumerate(ids) if pid in vectors]
        if present:
            stacked = np.stack([vectors[ids[i]] for i in present])
            centroid = stacked.mean(axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
            for i in present:
                raw[i] = float(vectors[ids[i]] @ centroid)
        if np.isnan(raw).all():
            return np.zeros(len(ids), dtype=np.float64)
        fill = float(np.nanmedian(raw))
        return np.where(np.isnan(raw), fill, raw)

    def _subject(self, ids: list[int], face_map: dict[int, list[tuple[float, bool]]]) -> _F:
        """Face presence weighted by quality, bonused for named people; capped at 1."""
        raw = np.zeros(len(ids), dtype=np.float64)
        for i, pid in enumerate(ids):
            total = 0.0
            for quality, named in face_map.get(pid, ()):
                total += quality * (1.0 + (_NAMED_BONUS if named else 0.0))
            raw[i] = min(1.0, total)
        return raw

    # ------------------------------------------------------------------ collapse
    def _collapse_keys(self, conn: Any, ids: list[int]) -> dict[int, tuple[str, int]]:
        """Cluster key per photo: its near-dup group, else burst group, else itself."""
        near_dup = groups_db.member_group_map(conn, "near_dup", ids)
        burst = groups_db.member_group_map(conn, "burst", ids)
        keys: dict[int, tuple[str, int]] = {}
        for pid in ids:
            if pid in near_dup:
                keys[pid] = ("near_dup", near_dup[pid])
            elif pid in burst:
                keys[pid] = ("burst", burst[pid])
            else:
                keys[pid] = ("single", pid)
        return keys

    # ------------------------------------------------------------------ driver
    def triage(
        self,
        conn: Any,
        *,
        preset: TriagePreset,
        group_id: int | None = None,
        date_from: float | None = None,
        date_to: float | None = None,
        limit: int = 50,
    ) -> TriageResult:
        ids = self._scope_ids(conn, group_id, date_from, date_to)
        rows = photos_db.triage_rows(conn, ids)
        ids = [pid for pid in ids if pid in rows]  # keep scope order, drop missing
        if not ids:
            return TriageResult(preset=preset.name, scope_size=0, considered=0, items=[])

        vectors = self._vectors(rows)
        face_map = faces_db.faces_with_person(conn, ids)

        quality_raw = np.array([rows[p]["quality"] or 0.0 for p in ids], dtype=np.float64)
        aesthetic_raw = np.array([rows[p]["aesthetic"] or 0.0 for p in ids], dtype=np.float64)
        repr_raw = self._representativeness(ids, vectors)
        subject_raw = self._subject(ids, face_map)

        norm = {
            "quality": _percentile_rank(quality_raw),
            "aesthetic": _percentile_rank(aesthetic_raw),
            "repr": _percentile_rank(repr_raw),
            "subject": _percentile_rank(subject_raw),
        }
        index = {pid: i for i, pid in enumerate(ids)}
        collapse = self._collapse_keys(conn, ids)

        if preset.invert:
            items = self._delete_candidates(ids, index, norm, collapse, limit)
            considered = len(items)
        else:
            base = (
                preset.weight_quality * norm["quality"]
                + preset.weight_aesthetic * norm["aesthetic"]
                + preset.weight_repr * norm["repr"]
                + preset.weight_subject * norm["subject"]
            )
            kept = self._collapse_best(ids, index, base, collapse)
            considered = len(kept)
            items = self._mmr_select(kept, index, base, norm, vectors, collapse, preset, limit)

        return TriageResult(
            preset=preset.name, scope_size=len(ids), considered=considered, items=items
        )

    def _scope_ids(
        self, conn: Any, group_id: int | None, date_from: float | None, date_to: float | None
    ) -> list[int]:
        if group_id is not None:
            group = groups_db.get_group(conn, group_id)
            if group is None:
                raise LookupError(f"group {group_id} not found")
            return groups_db.group_photo_ids(conn, group_id)
        if date_from is not None or date_to is not None:
            return photos_db.photos_in_range(conn, date_from, date_to)
        raise ValueError("triage needs a group_id or a date range")

    # ------------------------------------------------------------ keep presets
    def _collapse_best(
        self, ids: list[int], index: dict[int, int], base: _F, collapse: dict[int, tuple[str, int]]
    ) -> list[int]:
        """Keep the single best-scoring photo of each near-dup / burst cluster."""
        best: dict[tuple[str, int], int] = {}
        for pid in ids:
            key = collapse[pid]
            if key not in best or base[index[pid]] > base[index[best[key]]]:
                best[key] = pid
        # Preserve descending base-score order for a stable, meaningful shortlist.
        return sorted(best.values(), key=lambda p: base[index[p]], reverse=True)

    def _mmr_select(
        self,
        kept: list[int],
        index: dict[int, int],
        base: _F,
        norm: dict[str, _F],
        vectors: dict[int, _F],
        collapse: dict[int, tuple[str, int]],
        preset: TriagePreset,
        limit: int,
    ) -> list[TriageItem]:
        """Maximal-marginal-relevance pick: balance pick-score against visual diversity."""
        lam = preset.lam
        selected: list[int] = []
        remaining = list(kept)
        cluster_size: dict[tuple[str, int], int] = {}
        for pid in index:
            cluster_size[collapse[pid]] = cluster_size.get(collapse[pid], 0) + 1

        while remaining and len(selected) < limit:
            best_pid = None
            best_val = -np.inf
            for pid in remaining:
                sim = 0.0
                if pid in vectors and selected:
                    sims = [float(vectors[pid] @ vectors[s]) for s in selected if s in vectors]
                    sim = max(sims) if sims else 0.0
                val = lam * float(base[index[pid]]) - (1.0 - lam) * sim
                if val > best_val:
                    best_val, best_pid = val, pid
            assert best_pid is not None
            remaining.remove(best_pid)
            selected.append(best_pid)

        items: list[TriageItem] = []
        for pid in selected:
            i = index[pid]
            items.append(
                TriageItem(
                    photo_id=pid,
                    score=round(float(base[i]), 6),
                    quality=round(float(norm["quality"][i]), 6),
                    aesthetic=round(float(norm["aesthetic"][i]), 6),
                    representativeness=round(float(norm["repr"][i]), 6),
                    subject=round(float(norm["subject"][i]), 6),
                    reason=self._keep_reason(norm, i, collapse[pid], cluster_size),
                    redundant=False,
                )
            )
        return items

    def _keep_reason(
        self,
        norm: dict[str, _F],
        i: int,
        key: tuple[str, int],
        cluster_size: dict[tuple[str, int], int],
    ) -> str:
        parts: list[str] = []
        if norm["quality"][i] >= _HIGH:
            parts.append("sharp")
        if norm["aesthetic"][i] >= _HIGH:
            parts.append("vivid")
        if norm["repr"][i] >= _HIGH:
            parts.append("representative")
        if norm["subject"][i] >= _HIGH:
            parts.append("people")
        size = cluster_size.get(key, 1)
        if key[0] in ("near_dup", "burst") and size > 1:
            parts.append(f"best of {size} similar")
        return ", ".join(parts) if parts else "solid overall"

    # -------------------------------------------------------- delete candidates
    def _delete_candidates(
        self,
        ids: list[int],
        index: dict[int, int],
        norm: dict[str, _F],
        collapse: dict[int, tuple[str, int]],
        limit: int,
    ) -> list[TriageItem]:
        """Redundant (all-but-best of a cluster) and low-quality frames, worst first."""
        # Per cluster, the highest-quality frame is the keeper; the rest are redundant.
        keeper: dict[tuple[str, int], int] = {}
        for pid in ids:
            key = collapse[pid]
            if (
                key not in keeper
                or norm["quality"][index[pid]] > norm["quality"][index[keeper[key]]]
            ):
                keeper[key] = pid

        items: list[TriageItem] = []
        for pid in ids:
            i = index[pid]
            key = collapse[pid]
            redundant = key[0] in ("near_dup", "burst") and keeper[key] != pid
            badness = 0.5 * (1.0 - norm["quality"][i]) + 0.2 * (1.0 - norm["aesthetic"][i])
            if redundant:
                badness += _REDUNDANCY_BONUS
            elif norm["quality"][i] >= 0.5:
                continue  # a decent, non-redundant frame is not a delete candidate
            items.append(
                TriageItem(
                    photo_id=pid,
                    score=round(float(badness), 6),
                    quality=round(float(norm["quality"][i]), 6),
                    aesthetic=round(float(norm["aesthetic"][i]), 6),
                    representativeness=round(float(norm["repr"][i]), 6),
                    subject=round(float(norm["subject"][i]), 6),
                    reason=self._delete_reason(norm, i, redundant),
                    redundant=redundant,
                )
            )
        items.sort(key=lambda it: it.score, reverse=True)
        return items[:limit]

    def _delete_reason(self, norm: dict[str, _F], i: int, redundant: bool) -> str:
        parts: list[str] = []
        if redundant:
            parts.append("near-duplicate of a kept shot")
        if norm["quality"][i] < 0.34:
            parts.append("soft / low quality")
        if norm["aesthetic"][i] < 0.34:
            parts.append("flat")
        return ", ".join(parts) if parts else "weaker than the kept frame"
