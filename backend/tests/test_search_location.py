"""Date + location search: build_candidates GPS filter and the /search endpoint."""

from __future__ import annotations

import sqlite3
import time
from typing import Any, cast

import numpy as np
from fastapi.testclient import TestClient

from iris.config import Settings
from iris.db import apply_migrations, connect
from iris.db import photos as photos_db
from iris.geo.gazetteer import Gazetteer, GeoPlace
from iris.geo.service import GeoService
from iris.search import build_candidates


class _FakeEmbedder:
    def __init__(self, dim: int) -> None:
        self._vec = np.eye(1, dim, dtype=np.float32)[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec for _ in texts])


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _insert(
    conn: sqlite3.Connection, pid: int, *, sort_at: float, lat: float | None, lon: float | None
) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO photos (id, path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "gps_lat, gps_lon, thumb_at, missing, created_at, updated_at) "
        "VALUES (?, ?, '/s', ?, '.jpg', 1, ?, ?, ?, ?, ?, 0, ?, ?)",
        (pid, f"/s/p{pid}.jpg", f"p{pid}.jpg", now, sort_at, lat, lon, now, now, now),
    )


def test_build_candidates_gps_radius(settings: Settings) -> None:
    settings.ensure_dirs()
    conn = _open(settings)
    apply_migrations(conn)
    _insert(conn, 1, sort_at=1.0, lat=48.8566, lon=2.3522)  # central Paris
    _insert(conn, 2, sort_at=2.0, lat=48.80, lon=2.30)  # ~7 km south, still near
    _insert(conn, 3, sort_at=3.0, lat=51.5074, lon=-0.1278)  # London — far
    _insert(conn, 4, sort_at=4.0, lat=None, lon=None)  # no GPS

    near = build_candidates(conn, gps_center=(48.8534, 2.3488), gps_radius_km=25.0)
    assert near == {1, 2}
    conn.close()


def _seed_client(client: TestClient, settings: Settings) -> None:
    app = cast(Any, client.app)
    app.state.embeddings._embedder = _FakeEmbedder(settings.embed_dim)
    app.state.geo = GeoService(settings, gazetteer=Gazetteer([_PARIS], {"paris": [0]}))
    conn = _open(settings)
    _insert(conn, 1, sort_at=_ts(2024, 6, 1), lat=48.8566, lon=2.3522)  # Paris, 2024
    _insert(conn, 2, sort_at=_ts(2020, 6, 1), lat=48.8566, lon=2.3522)  # Paris, 2020
    _insert(conn, 3, sort_at=_ts(2024, 6, 1), lat=51.5074, lon=-0.1278)  # London, 2024
    vectors = np.eye(3, settings.embed_dim, dtype=np.float32)
    start = app.state.embeddings.store.append(vectors)
    conn.execute("BEGIN")
    for i in range(3):
        photos_db.set_embed(conn, i + 1, start + i, time.time())
    conn.execute("COMMIT")
    conn.close()
    app.state.embeddings.ensure_index()


_PARIS = GeoPlace("Paris", "FR", 48.8534, 2.3488, 2_138_551)


def _ts(y: int, m: int, d: int) -> float:
    return time.mktime((y, m, d, 12, 0, 0, 0, 0, -1))


def test_search_filter_only_by_year(client: TestClient, settings: Settings) -> None:
    _seed_client(client, settings)
    body = client.post("/search", json={"query": "2024", "limit": 10}).json()
    assert body["tier"] == "filter"
    assert body["applied"]["date_label"] == "2024"
    assert {i["id"] for i in body["items"]} == {1, 3}  # both 2024 photos, any location


def test_search_place_filter(client: TestClient, settings: Settings) -> None:
    _seed_client(client, settings)
    body = client.post(
        "/search", json={"query": "2024", "limit": 10, "filters": {"place": "Paris"}}
    ).json()
    assert body["applied"]["place_label"] == "Paris, FR"
    assert {i["id"] for i in body["items"]} == {1}  # Paris + 2024 only


def test_search_unknown_place_reports_error(client: TestClient, settings: Settings) -> None:
    _seed_client(client, settings)
    body = client.post(
        "/search", json={"query": "dog", "limit": 10, "filters": {"place": "Atlantis"}}
    ).json()
    assert body["place_error"] is not None
