"""People / faces API tests (seeded through the app's own FacesService)."""

from __future__ import annotations

import sqlite3
import time

import numpy as np
from fastapi.testclient import TestClient

from iris.app import create_app
from iris.config import Settings
from iris.db import connect
from iris.db import faces as faces_db

DIM = 512


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _insert_photo(conn: sqlite3.Connection) -> int:
    now = time.time()
    cur = conn.execute(
        "INSERT INTO photos (path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "missing, thumb_at, content_hash, created_at, updated_at) "
        "VALUES (?, '/s', 'p.jpg', '.jpg', 1, ?, ?, 0, ?, 'h', ?, ?)",
        (f"/s/{time.time_ns()}.jpg", now, now, now, now, now),
    )
    assert cur.lastrowid is not None
    return cur.lastrowid


def _unit(center: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.zeros(DIM, dtype=np.float32)
    v[center] = 1.0
    v = v + 0.02 * rng.standard_normal(DIM).astype(np.float32)
    return np.asarray(v / np.linalg.norm(v), dtype=np.float32)


def test_people_api_flow(settings: Settings) -> None:
    settings.ensure_dirs()
    app = create_app(settings)
    with TestClient(app) as client:
        service = app.state.faces  # the real FacesService (shares its memmap store)
        conn = _open(settings)
        p1, p2 = _insert_photo(conn), _insert_photo(conn)
        # group A on p1, group B on p2 (2 faces each, well-separated)
        vectors = np.stack([_unit(0, 1), _unit(0, 2), _unit(1, 3), _unit(1, 4)])
        start = service.store.append(vectors)
        conn.execute("BEGIN")
        for i, photo_id in enumerate([p1, p1, p2, p2]):
            faces_db.insert_face(
                conn,
                photo_id=photo_id,
                bbox=(0.1, 0.1, 0.2, 0.2),
                det_score=0.9,
                landmarks=None,
                quality=0.5 + i * 0.01,
                embed_row=start + i,
                now=time.time(),
            )
        conn.execute("COMMIT")
        service.update_people(conn)  # forms 2 people from the pending pool
        conn.close()

        people = client.get("/people").json()
        assert len(people) == 2
        person_id = people[0]["id"]

        detail = client.get(f"/people/{person_id}").json()
        assert detail["person"]["id"] == person_id
        assert len(detail["photos"]) >= 1

        # rename pins the person
        renamed = client.patch(f"/people/{person_id}", json={"label": "Alice"}).json()
        assert renamed["label"] == "Alice"
        assert renamed["pinned"] == 1

        # faces for a photo
        faces = client.get(f"/photos/{p1}/faces").json()
        assert len(faces) == 2

        # merge the two people into one
        ids = [p["id"] for p in people]
        merged = client.post("/people/merge", json={"ids": ids})
        assert merged.status_code == 200
        assert len(client.get("/people").json()) == 1
        assert merged.json()["size"] == 4
