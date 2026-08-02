"""/triage + /triage/presets endpoint tests (real EmbeddingService store, no CLIP)."""

from __future__ import annotations

import sqlite3
import time
from typing import Any, cast

import numpy as np
from fastapi.testclient import TestClient

from iris.config import Settings
from iris.db import connect
from iris.db.groups import GroupSpec, replace_groups


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _seed(conn: sqlite3.Connection, pid: int, aesthetic: float, quality: float, row: int) -> None:
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
            row,
            now,
            now,
            now,
            now,
        ),
    )


def _seed_event(client: TestClient, settings: Settings) -> int:
    service = cast(Any, client.app).state.embeddings
    vecs = np.eye(3, settings.embed_dim, dtype=np.float32)
    start = service.store.append(vecs)
    conn = _open(settings)
    _seed(conn, 1, 0.9, 0.9, start)
    _seed(conn, 2, 0.5, 0.5, start + 1)
    _seed(conn, 3, 0.2, 0.2, start + 2)
    replace_groups(conn, "event", [GroupSpec(kind="event", members=[(1, 0.0), (2, 1.0), (3, 2.0)])])
    event_id = int(conn.execute("SELECT id FROM groups WHERE kind = 'event'").fetchone()[0])
    conn.close()
    return event_id


def test_presets_listed(client: TestClient) -> None:
    resp = client.get("/triage/presets")
    assert resp.status_code == 200
    names = {p["name"] for p in resp.json()}
    assert names == {"story-ready", "print-worthy", "delete-candidates"}


def test_triage_group(client: TestClient, settings: Settings) -> None:
    event_id = _seed_event(client, settings)
    resp = client.post("/triage", json={"group_id": event_id, "preset": "story-ready"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["preset"] == "story-ready"
    assert body["scope_size"] == 3
    ids = [it["id"] for it in body["items"]]
    assert ids[0] == 1  # highest all-round score leads
    first = body["items"][0]
    assert set(first) >= {"quality", "aesthetic", "representativeness", "subject", "reason"}


def test_triage_unknown_preset(client: TestClient, settings: Settings) -> None:
    event_id = _seed_event(client, settings)
    resp = client.post("/triage", json={"group_id": event_id, "preset": "nope"})
    assert resp.status_code == 422


def test_triage_missing_group(client: TestClient) -> None:
    resp = client.post("/triage", json={"group_id": 99999, "preset": "story-ready"})
    assert resp.status_code == 404


def test_triage_requires_scope(client: TestClient) -> None:
    resp = client.post("/triage", json={"preset": "story-ready"})
    assert resp.status_code == 422
