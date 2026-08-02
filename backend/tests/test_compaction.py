"""Memmap compaction: reclaim orphaned vectors + renumber DB pointers (clip & faces)."""

from __future__ import annotations

import sqlite3
import time

import numpy as np

from iris.config import Settings
from iris.db import apply_migrations, connect
from iris.db import faces as faces_db
from iris.embeddings.service import EmbeddingService
from iris.faces.service import FacesService


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _seed_photo(conn: sqlite3.Connection, pid: int, embed_row: int, *, missing: int = 0) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO photos (id, path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "embed_row, embed_at, missing, created_at, updated_at) "
        "VALUES (?, ?, '/s', ?, '.jpg', 1, ?, ?, ?, ?, ?, ?, ?)",
        (pid, f"/s/p{pid}.jpg", f"p{pid}.jpg", now, float(pid), embed_row, now, missing, now, now),
    )


def test_clip_compaction_reclaims_and_renumbers(settings: Settings) -> None:
    settings.ensure_dirs()
    service = EmbeddingService(settings)
    dim = settings.embed_dim
    vecs = np.eye(4, dim, dtype=np.float32)  # rows 0..3 = one-hot columns 0..3
    service.store.append(vecs)

    conn = _open(settings)
    apply_migrations(conn)
    _seed_photo(conn, 0, 0)
    _seed_photo(conn, 1, 1, missing=1)  # soft-deleted -> row 1 becomes an orphan
    _seed_photo(conn, 2, 2)
    _seed_photo(conn, 3, 3)
    conn.commit()
    conn.close()

    stats = service.compact()
    assert stats == {"before": 4, "after": 3, "reclaimed": 1}
    assert service.store.count == 3

    conn = _open(settings)
    rows = {r[0]: r[1] for r in conn.execute("SELECT id, embed_row FROM photos").fetchall()}
    conn.close()
    assert rows[1] is None  # dangling pointer cleared on the missing photo
    # Live photos renumbered to a dense 0..2 range, order preserved.
    assert sorted(v for v in rows.values() if v is not None) == [0, 1, 2]
    # Vectors survived: photo 3's stored vector is still one-hot column 3.
    vec3 = service.get_rows([rows[3]])[0]
    assert int(np.argmax(vec3)) == 3
    # Index rebuilt from the compacted store: querying photo 3's vector returns photo 3.
    ids, _ = service.query(np.eye(1, dim, k=3, dtype=np.float32), k=1)
    assert ids[0] == 3


def test_faces_compaction_renumbers(settings: Settings) -> None:
    settings.ensure_dirs()
    faces = FacesService(settings)
    faces.store.append(np.eye(3, settings.face_dim, dtype=np.float32))

    conn = _open(settings)
    apply_migrations(conn)
    _seed_photo(conn, 0, 0)  # a host photo for the FK
    now = time.time()
    f0 = faces_db.insert_face(
        conn,
        photo_id=0,
        bbox=(0, 0, 1, 1),
        det_score=0.9,
        landmarks=None,
        quality=0.5,
        embed_row=0,
        now=now,
    )
    f1 = faces_db.insert_face(
        conn,
        photo_id=0,
        bbox=(0, 0, 1, 1),
        det_score=0.9,
        landmarks=None,
        quality=0.5,
        embed_row=1,
        now=now,
    )
    f2 = faces_db.insert_face(
        conn,
        photo_id=0,
        bbox=(0, 0, 1, 1),
        det_score=0.9,
        landmarks=None,
        quality=0.5,
        embed_row=2,
        now=now,
    )
    conn.execute("DELETE FROM faces WHERE id = ?", (f1,))  # row 1 orphaned
    conn.commit()

    stats = faces.compact(conn)
    assert stats["before"] == 3 and stats["after"] == 2 and stats["reclaimed"] == 1
    assert faces.store.count == 2
    rows = {r[0]: r[1] for r in conn.execute("SELECT id, embed_row FROM faces").fetchall()}
    conn.close()
    assert sorted(rows.values()) == [0, 1]  # f0, f2 renumbered to a dense range
    assert rows[f0] == 0 and rows[f2] == 1
    _ = f2  # (silence unused in case of future edits)


def test_compaction_noop_when_dense(settings: Settings) -> None:
    settings.ensure_dirs()
    service = EmbeddingService(settings)
    service.store.append(np.eye(2, settings.embed_dim, dtype=np.float32))
    conn = _open(settings)
    apply_migrations(conn)
    _seed_photo(conn, 0, 0)
    _seed_photo(conn, 1, 1)
    conn.commit()
    conn.close()
    stats = service.compact()
    assert stats == {"before": 2, "after": 2, "reclaimed": 0}
