"""TriageService: four-score normalization, near-dup collapse, MMR, delete-candidates.

Uses a real :class:`EmbeddingService` (vectors appended to its memmap store) and a real
SQLite DB — no CLIP model is loaded, since triage only *reads* stored vectors.
"""

from __future__ import annotations

import sqlite3
import time

import numpy as np

from iris.config import Settings
from iris.db import apply_migrations, connect
from iris.db.groups import GroupSpec, replace_groups
from iris.embeddings.service import EmbeddingService
from iris.triage import get_preset
from iris.triage.service import TriageService, _percentile_rank


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _seed_photo(
    conn: sqlite3.Connection, pid: int, *, aesthetic: float, quality: float, embed_row: int
) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO photos (id, path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "aesthetic, quality, embed_row, phash, thumb_at, scored_at, missing, created_at, "
        "updated_at) VALUES (?, ?, '/s', ?, '.jpg', 1, ?, ?, ?, ?, ?, 0, ?, ?, 0, ?, ?)",
        (
            pid,
            f"/s/p{pid}.jpg",
            f"p{pid}.jpg",
            now,
            float(pid),
            aesthetic,
            quality,
            embed_row,
            now,
            now,
            now,
            now,
        ),
    )


def _setup(settings: Settings) -> tuple[EmbeddingService, int]:
    """Five photos in one event; p1&p2 are near-duplicates (p1 sharp, p2 blurry).

    Vectors: p1,p2,p3,p5 point along e0 (the theme); p4 points along e1 (an outlier that
    adds diversity and has low representativeness). Returns (service, event_group_id).
    """
    dim = settings.embed_dim
    service = EmbeddingService(settings)
    vecs = np.zeros((5, dim), dtype=np.float32)
    for i in (0, 1, 2, 4):  # p1,p2,p3,p5 -> e0
        vecs[i, 0] = 1.0
    vecs[3, 1] = 1.0  # p4 -> e1 (orthogonal)
    start = service.store.append(vecs)

    conn = _open(settings)
    apply_migrations(conn)
    _seed_photo(conn, 1, aesthetic=0.9, quality=0.9, embed_row=start + 0)
    _seed_photo(conn, 2, aesthetic=0.1, quality=0.1, embed_row=start + 1)
    _seed_photo(conn, 3, aesthetic=0.5, quality=0.5, embed_row=start + 2)
    _seed_photo(conn, 4, aesthetic=0.5, quality=0.5, embed_row=start + 3)
    _seed_photo(conn, 5, aesthetic=0.5, quality=0.5, embed_row=start + 4)

    members = [(pid, float(i)) for i, pid in enumerate([1, 2, 3, 4, 5])]
    replace_groups(conn, "event", [GroupSpec(kind="event", members=members)])
    replace_groups(conn, "near_dup", [GroupSpec(kind="near_dup", members=[(1, 0.0), (2, 1.0)])])
    event_id = int(conn.execute("SELECT id FROM groups WHERE kind = 'event'").fetchone()[0])
    conn.close()
    return service, event_id


def test_percentile_rank_basic() -> None:
    np.testing.assert_allclose(_percentile_rank(np.array([10.0, 20.0, 30.0])), [0.0, 0.5, 1.0])
    # ties share the average rank
    np.testing.assert_allclose(_percentile_rank(np.array([5.0, 5.0, 10.0])), [0.25, 0.25, 1.0])
    assert _percentile_rank(np.array([7.0]))[0] == 0.5  # single element -> neutral


def test_near_dup_collapse_keeps_best(settings: Settings) -> None:
    service, event_id = _setup(settings)
    conn = _open(settings)
    result = TriageService(settings, service).triage(
        conn, preset=get_preset("story-ready"), group_id=event_id, limit=10
    )
    conn.close()
    ids = [it.photo_id for it in result.items]
    assert result.scope_size == 5
    assert result.considered == 4  # p1&p2 collapsed to one
    assert 2 not in ids  # the blurry near-duplicate is dropped
    assert 1 in ids  # the sharp one is kept
    for it in result.items:  # every normalized score stays in range
        assert 0.0 <= it.quality <= 1.0 and 0.0 <= it.representativeness <= 1.0


def test_top_pick_is_the_hero(settings: Settings) -> None:
    service, event_id = _setup(settings)
    conn = _open(settings)
    for preset in ("story-ready", "print-worthy"):
        result = TriageService(settings, service).triage(
            conn, preset=get_preset(preset), group_id=event_id, limit=10
        )
        assert result.items[0].photo_id == 1, preset  # p1 dominates every score
    conn.close()


def test_mmr_surfaces_the_diverse_outlier(settings: Settings) -> None:
    service, event_id = _setup(settings)
    conn = _open(settings)
    # With only 2 slots, the diversity-leaning preset should pair the hero with the
    # orthogonal outlier (p4) rather than another near-identical e0 shot.
    story = TriageService(settings, service).triage(
        conn, preset=get_preset("story-ready"), group_id=event_id, limit=2
    )
    conn.close()
    ids = [it.photo_id for it in story.items]
    assert ids[0] == 1
    assert 4 in ids  # MMR chose the diverse frame for the second slot


def test_delete_candidates_flags_redundant(settings: Settings) -> None:
    service, event_id = _setup(settings)
    conn = _open(settings)
    result = TriageService(settings, service).triage(
        conn, preset=get_preset("delete-candidates"), group_id=event_id, limit=10
    )
    conn.close()
    by_id = {it.photo_id: it for it in result.items}
    assert 2 in by_id and by_id[2].redundant is True  # blurry near-dup surfaces
    assert by_id[2].photo_id == result.items[0].photo_id  # worst-first ordering
    assert 1 not in by_id  # the kept hero is not a delete candidate


def test_date_range_scope(settings: Settings) -> None:
    service, _ = _setup(settings)
    conn = _open(settings)
    result = TriageService(settings, service).triage(
        conn, preset=get_preset("story-ready"), date_from=0.0, date_to=1e12, limit=10
    )
    conn.close()
    assert result.scope_size == 5  # all five photos fall in the range


def test_deterministic(settings: Settings) -> None:
    service, event_id = _setup(settings)
    conn = _open(settings)
    svc = TriageService(settings, service)
    a = svc.triage(conn, preset=get_preset("story-ready"), group_id=event_id, limit=10)
    b = svc.triage(conn, preset=get_preset("story-ready"), group_id=event_id, limit=10)
    conn.close()
    assert [i.photo_id for i in a.items] == [i.photo_id for i in b.items]
