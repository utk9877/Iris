"""Integration test for the Phase 1 API: roots -> scan -> photos -> thumb."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image


def _wait_ingest_done(client: TestClient, timeout: float = 90.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body: dict[str, Any] = client.get("/ingest/status").json()
        if (
            not body["running"]
            and body["job"]
            and body["job"]["state"] in {"done", "error", "canceled"}
        ):
            return body
        time.sleep(0.1)
    raise AssertionError("ingest did not finish")


def test_ingest_flow(client: TestClient, tmp_path: Path) -> None:
    src = tmp_path / "photos"
    src.mkdir()
    for i, color in enumerate([(200, 30, 30), (30, 200, 30), (30, 30, 200)]):
        Image.new("RGB", (160, 120), color).save(src / f"p{i}.jpg")

    # Add a root.
    resp = client.post("/library/roots", json={"path": str(src)})
    assert resp.status_code == 200
    assert client.get("/library/roots").json()[0]["path"] == str(src.resolve())

    # Reject a non-directory.
    assert client.post("/library/roots", json={"path": str(src / "nope")}).status_code == 422

    # Scan and wait.
    assert client.post("/ingest/scan").json()["job_id"] >= 1
    final = _wait_ingest_done(client)
    assert final["job"]["state"] == "done"

    # Count + list.
    assert client.get("/photos/count").json()["count"] == 3
    page = client.get("/photos", params={"limit": 2}).json()
    assert len(page["items"]) == 2
    assert page["next_cursor"] is not None
    assert all(item["has_thumb"] for item in page["items"])

    # Second page via cursor; no third page.
    page2 = client.get("/photos", params={"limit": 2, "cursor": page["next_cursor"]}).json()
    assert len(page2["items"]) == 1
    assert page2["next_cursor"] is None

    # Thumbnail bytes are served as webp.
    photo_id = page["items"][0]["id"]
    thumb = client.get(f"/thumb/{photo_id}")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/webp"
    assert len(thumb.content) > 0

    # /meta now reflects 3 indexed photos.
    assert client.get("/meta").json()["counts"]["photos"] == 3
