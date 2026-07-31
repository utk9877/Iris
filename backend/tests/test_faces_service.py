"""FacesService tests: initial clustering + incremental assignment (synthetic faces)."""

from __future__ import annotations

import sqlite3
import time

import numpy as np

from iris.config import Settings
from iris.db import apply_migrations, connect
from iris.db import faces as faces_db
from iris.faces.service import FacesService

DIM = 512
GROUPS, PER = 3, 8


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _insert_photo(conn: sqlite3.Connection) -> int:
    now = time.time()
    cur = conn.execute(
        "INSERT INTO photos (path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "missing, created_at, updated_at) VALUES (?, '/s', 'p.jpg', '.jpg', 1, ?, ?, 0, ?, ?)",
        (f"/s/p{time.time_ns()}.jpg", now, now, now, now),
    )
    assert cur.lastrowid is not None
    return cur.lastrowid


def _centers(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((GROUPS, DIM)).astype(np.float32)
    return np.asarray(c / np.linalg.norm(c, axis=1, keepdims=True), dtype=np.float32)


def _near(center: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = center + 0.02 * rng.standard_normal(DIM).astype(np.float32)
    return np.asarray(v / np.linalg.norm(v), dtype=np.float32)


def _seed(settings: Settings) -> tuple[FacesService, np.ndarray]:
    settings.ensure_dirs()
    conn = _open(settings)
    apply_migrations(conn)
    photo_id = _insert_photo(conn)
    service = FacesService(settings)
    centers = _centers()

    vectors = np.stack(
        [_near(centers[g], seed=1000 * g + i) for g in range(GROUPS) for i in range(PER)]
    )
    start = service.store.append(vectors)
    conn.execute("BEGIN")
    for offset in range(GROUPS * PER):
        faces_db.insert_face(
            conn,
            photo_id=photo_id,
            bbox=(0.0, 0.0, 0.1, 0.1),
            det_score=0.9,
            landmarks=None,
            quality=0.5 + offset * 0.001,
            embed_row=start + offset,
            now=time.time(),
        )
    conn.execute("COMMIT")
    conn.close()
    return service, centers


def test_initial_clustering_forms_people(settings: Settings) -> None:
    service, _ = _seed(settings)
    conn = _open(settings)
    try:
        service.update_people(conn)  # no clusters yet -> reclusters all pending
        people = faces_db.list_people(conn)
        assert len(people) == GROUPS
        assert sorted(p["size"] for p in people) == [PER, PER, PER]
        assert faces_db.count_pending_pool(conn) == 0  # everyone assigned
        for person in people:
            assert person["rep_face_id"] is not None
    finally:
        conn.close()


def test_incremental_assignment_joins_existing_person(settings: Settings) -> None:
    service, centers = _seed(settings)
    conn = _open(settings)
    try:
        service.update_people(conn)
        before = {p["id"]: p["size"] for p in faces_db.list_people(conn)}

        # A new face near group 0 arrives (assigned=0).
        new_vec = _near(centers[0], seed=99999).reshape(1, DIM)
        row = service.store.append(new_vec)
        photo_id = _insert_photo(conn)
        faces_db.insert_face(
            conn,
            photo_id=photo_id,
            bbox=(0.0, 0.0, 0.1, 0.1),
            det_score=0.95,
            landmarks=None,
            quality=0.9,
            embed_row=row,
            now=time.time(),
        )

        assigned = service.assign_pending(conn)
        assert assigned == 1  # joined an existing person by centroid, no re-cluster
        after = {p["id"]: p["size"] for p in faces_db.list_people(conn)}
        assert len(after) == len(before)  # no new person created
        assert sum(after.values()) == sum(before.values()) + 1
    finally:
        conn.close()


def test_force_recluster_preserves_pinned_people(settings: Settings) -> None:
    service, _ = _seed(settings)
    conn = _open(settings)
    try:
        service.update_people(conn)
        people = faces_db.list_people(conn)
        assert len(people) == GROUPS
        pinned_id = people[0]["id"]
        faces_db.set_label(conn, pinned_id, "Alice", time.time())

        service.update_people(conn, force=True)  # dissolves unlabeled, keeps Alice
        after = {p["id"]: p["label"] for p in faces_db.list_people(conn)}
        assert pinned_id in after  # the labeled person kept its identity
        assert after[pinned_id] == "Alice"
        assert len(after) == GROUPS  # re-cluster reproduced the other people
    finally:
        conn.close()
