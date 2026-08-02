"""/maintenance/storage + /maintenance/compact endpoint tests."""

from __future__ import annotations

from typing import Any, cast

from fastapi.testclient import TestClient


def test_storage_stats_shape(client: TestClient) -> None:
    body = client.get("/maintenance/storage").json()
    assert set(body) == {"thumbs", "previews", "embeddings", "database"}
    for area in body.values():
        assert area["bytes"] >= 0
        assert "over_cap" in area
    # Capped areas report a cap; embeddings/db are uncapped.
    assert body["previews"]["cap_bytes"] is not None
    assert body["embeddings"]["cap_bytes"] is None


def test_compact_empty_is_noop(client: TestClient) -> None:
    body = client.post("/maintenance/compact").json()
    assert body["clip"] == {"before": 0, "after": 0, "reclaimed": 0}
    assert body["faces"] == {"before": 0, "after": 0, "reclaimed": 0}


def test_compact_conflicts_with_running_ingest(client: TestClient) -> None:
    class _BusyIngest:
        def is_running(self) -> bool:
            return True

    cast(Any, client.app).state.ingest = _BusyIngest()
    resp = client.post("/maintenance/compact")
    assert resp.status_code == 409
