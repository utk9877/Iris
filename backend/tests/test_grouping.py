"""Grouping unit tests (events/bursts/near-dups/themes) + a rebuild integration."""

from __future__ import annotations

import sqlite3
import time

import numpy as np

from iris.config import Settings
from iris.db import apply_migrations, connect
from iris.db import groups as groups_db
from iris.db import photos as photos_db
from iris.embeddings.service import EmbeddingService
from iris.grouping.events import detect_bursts, segment_events
from iris.grouping.near_dup import find_near_dups, hamming64
from iris.grouping.semantic import cluster_semantic
from iris.grouping.service import GroupingService


def _row(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": 0,
        "sort_at": 0.0,
        "taken_at": None,
        "camera_model": "Cam",
        "phash": 0,
        "embed_row": None,
        "gps_lat": None,
        "gps_lon": None,
        "width": 100,
        "height": 100,
    }
    base.update(kw)
    return base


# --- events / bursts ---


def test_segment_events_splits_on_time_gap() -> None:
    rows = [_row(id=i, sort_at=float(i)) for i in range(3)]
    rows += [_row(id=i, sort_at=100000.0 + i) for i in range(3, 6)]
    events = segment_events(rows, gap_seconds=3600, gps_km=50)
    assert [len(e) for e in events] == [3, 3]


def test_segment_events_splits_on_gps_jump() -> None:
    rows = [
        _row(id=0, sort_at=0.0, gps_lat=40.0, gps_lon=-74.0),
        _row(id=1, sort_at=1.0, gps_lat=40.0, gps_lon=-74.0),
        _row(id=2, sort_at=2.0, gps_lat=48.85, gps_lon=2.35),  # Paris — huge jump
    ]
    events = segment_events(rows, gap_seconds=3600, gps_km=50)
    assert [len(e) for e in events] == [2, 1]


def test_detect_bursts_needs_same_camera_and_small_gap() -> None:
    rows = [
        _row(id=0, sort_at=0.0, camera_model="A"),
        _row(id=1, sort_at=0.5, camera_model="A"),  # burst with 0
        _row(id=2, sort_at=1.0, camera_model="B"),  # camera change breaks it
        _row(id=3, sort_at=60.0, camera_model="B"),  # gap too large
    ]
    bursts = detect_bursts(rows, gap_seconds=2.0)
    assert len(bursts) == 1
    assert [r["id"] for r in bursts[0]] == [0, 1]


# --- near-dup ---


def test_hamming64_handles_negative_signed_ints() -> None:
    assert hamming64(0, 0) == 0
    assert hamming64(0, 1) == 1
    assert hamming64(-1, 0) == 64  # -1 == all bits set as unsigned 64-bit


def test_find_near_dups_groups_close_hashes() -> None:
    rows = [
        _row(id=0, phash=0b0000),
        _row(id=1, phash=0b0001),  # 1 bit from id 0
        _row(id=2, phash=0xFFFFFF),  # 24 bits set -> far from both
    ]
    comps = find_near_dups(rows, hamming_max=2, cosine_min=0.0)
    assert len(comps) == 1
    assert {r["id"] for r in comps[0]} == {0, 1}


def test_find_near_dups_cosine_rejects_phash_collision() -> None:
    rows = [
        _row(id=0, phash=0b0000, embed_row=0),
        _row(id=1, phash=0b0001, embed_row=1),
    ]
    orthogonal = {0: np.array([1.0, 0.0], np.float32), 1: np.array([0.0, 1.0], np.float32)}
    comps = find_near_dups(rows, hamming_max=2, cosine_min=0.9, vectors=orthogonal)
    assert comps == []  # phash-close but cosine 0.0 < 0.9 -> not a duplicate


# --- semantic ---


def test_cluster_semantic_finds_two_themes() -> None:
    a = np.tile(np.array([1.0, 0.0], np.float32), (6, 1))
    b = np.tile(np.array([0.0, 1.0], np.float32), (6, 1))
    matrix = np.vstack([a, b]).astype(np.float32)
    ids = list(range(12))
    themes = cluster_semantic(ids, matrix, k=5, edge_threshold=0.5, min_size=3)
    assert len(themes) == 2
    assert all(len(t.members) == 6 for t in themes)


# --- rebuild integration ---


def _insert(conn: sqlite3.Connection, pid: int, **kw: object) -> None:
    now = time.time()
    cols = _row(id=pid, **kw)
    conn.execute(
        "INSERT INTO photos (id, path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "taken_at, camera_model, phash, width, height, gps_lat, gps_lon, "
        "phash_at, thumb_at, missing, created_at, updated_at) "
        "VALUES (?, ?, '/s', ?, '.jpg', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
        (
            pid,
            f"/s/p{pid}.jpg",
            f"p{pid}.jpg",
            now,
            cols["sort_at"],
            cols["taken_at"],
            cols["camera_model"],
            cols["phash"],
            cols["width"],
            cols["height"],
            cols["gps_lat"],
            cols["gps_lon"],
            now,
            now,
            now,
            now,
        ),
    )


def test_rebuild_writes_all_layers(settings: Settings) -> None:
    settings.ensure_dirs()
    conn = connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    apply_migrations(conn)

    # Event A: a 3-shot burst, two of them exact near-duplicates.
    _insert(conn, 0, sort_at=1000.0, phash=0b1010, embed_row=0)
    _insert(conn, 1, sort_at=1001.0, phash=0b1010, embed_row=1)  # dup of 0
    _insert(conn, 2, sort_at=1001.5, phash=(1 << 50), embed_row=2)
    # Event B: much later, distinct look.
    _insert(conn, 3, sort_at=500000.0, phash=(1 << 20), embed_row=3)
    _insert(conn, 4, sort_at=500001.0, phash=(1 << 21), embed_row=4)
    conn.execute("BEGIN")
    for pid in range(5):
        photos_db.set_embed(conn, pid, pid, time.time())
    conn.execute("COMMIT")

    service = EmbeddingService(settings)
    # Two exact-dup vectors (rows 0,1) + distinct others -> near-dup cosine confirms.
    vectors = np.eye(5, settings.embed_dim, dtype=np.float32)
    vectors[1] = vectors[0]
    service.store.append(vectors)

    grouping = GroupingService(settings, service)
    counts = grouping.rebuild(conn)

    assert counts["event"] == 2  # A and B
    assert counts["burst"] >= 1  # the 3-shot run in event A
    assert counts["near_dup"] == 1
    dup = groups_db.list_groups(conn, kind="near_dup", limit=10)[0]
    assert dup["size"] == 2
    assert set(groups_db.group_photo_ids(conn, dup["id"])) == {0, 1}
    conn.close()
