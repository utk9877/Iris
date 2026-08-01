"""API tests for the groups endpoints."""

from __future__ import annotations

import sqlite3
import time

from fastapi.testclient import TestClient

from iris.config import Settings
from iris.db import connect
from iris.db import groups as groups_db
from iris.db.groups import GroupSpec


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _seed_photo(conn: sqlite3.Connection, pid: int) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO photos (id, path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "thumb_at, missing, created_at, updated_at) "
        "VALUES (?, ?, '/s', ?, '.jpg', 1, ?, ?, ?, 0, ?, ?)",
        (pid, f"/s/p{pid}.jpg", f"p{pid}.jpg", now, now, now, now, now),
    )


def test_list_and_get_groups(client: TestClient, settings: Settings) -> None:
    conn = _open(settings)
    for pid in (1, 2, 3):
        _seed_photo(conn, pid)
    groups_db.replace_groups(
        conn,
        "event",
        [
            GroupSpec(
                kind="event",
                key="2026-07-01",
                rep_photo_id=1,
                start_at=1000.0,
                end_at=1002.0,
                members=[(1, 0.0), (2, 1.0), (3, 2.0)],
            )
        ],
    )
    conn.close()

    resp = client.get("/groups", params={"kind": "event"})
    assert resp.status_code == 200
    groups = resp.json()
    assert len(groups) == 1
    assert groups[0]["size"] == 3
    assert groups[0]["key"] == "2026-07-01"

    gid = groups[0]["id"]
    detail = client.get(f"/groups/{gid}").json()
    assert [p["id"] for p in detail["photos"]] == [1, 2, 3]  # rank order


def test_groups_rejects_unknown_kind(client: TestClient) -> None:
    assert client.get("/groups", params={"kind": "bogus"}).status_code == 422


def test_get_missing_group_404(client: TestClient) -> None:
    assert client.get("/groups/999").status_code == 404
