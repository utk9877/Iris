"""API tests for tags CRUD and the per-photo OCR endpoint."""

from __future__ import annotations

import sqlite3
import time

from fastapi.testclient import TestClient

from iris.config import Settings
from iris.db import connect
from iris.db import ocr as ocr_db


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _seed_photo(conn: sqlite3.Connection, pid: int) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO photos (id, path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "missing, created_at, updated_at) VALUES (?, ?, '/s', ?, '.jpg', 1, ?, ?, 0, ?, ?)",
        (pid, f"/s/p{pid}.jpg", f"p{pid}.jpg", now, now, now, now),
    )


def test_tag_lifecycle(client: TestClient, settings: Settings) -> None:
    conn = _open(settings)
    _seed_photo(conn, 1)
    conn.close()

    created = client.post("/photos/1/tags", json={"name": "sunset"})
    assert created.status_code == 200
    tag_id = created.json()["id"]

    # idempotent re-add + listing
    client.post("/photos/1/tags", json={"name": "sunset"})
    assert [t["name"] for t in client.get("/photos/1/tags").json()] == ["sunset"]
    tags = client.get("/tags").json()
    assert tags[0]["name"] == "sunset" and tags[0]["count"] == 1

    # removal
    assert client.delete(f"/photos/1/tags/{tag_id}").status_code == 200
    assert client.get("/photos/1/tags").json() == []


def test_add_tag_validations(client: TestClient, settings: Settings) -> None:
    conn = _open(settings)
    _seed_photo(conn, 1)
    conn.close()
    assert client.post("/photos/1/tags", json={"name": "  "}).status_code == 422
    assert client.post("/photos/999/tags", json={"name": "x"}).status_code == 404


def test_photo_ocr_endpoint(client: TestClient, settings: Settings) -> None:
    conn = _open(settings)
    _seed_photo(conn, 1)
    ocr_db.set_ocr(
        conn,
        photo_id=1,
        text="hello world",
        regions=[(0.0, 0.0, 0.3, 0.1, 0.9, "hello world")],
        now=time.time(),
    )
    conn.close()

    data = client.get("/photos/1/ocr").json()
    assert data["text"] == "hello world"
    assert data["regions"][0]["text"] == "hello world"
    # a photo with no OCR yields empty (not 404)
    _open(settings).close()
    assert client.get("/photos/2/ocr").json() == {"photo_id": 2, "text": "", "regions": []}
