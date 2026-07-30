"""System API + benchmark harness integration tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from iris.db import latest_version


def test_index_returns_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Iris" in resp.text


def test_cors_allows_dev_origin(client: TestClient) -> None:
    """The Tauri dev webview (localhost:1420) must be allowed to fetch the sidecar."""
    resp = client.get("/health", headers={"Origin": "http://localhost:1420"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:1420"


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_meta(client: TestClient) -> None:
    resp = client.get("/meta")
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == latest_version()
    # Fresh library: schema exists but nothing ingested yet.
    assert body["counts"]["photos"] == 0
    assert "benchmarks" in body["counts"]


def test_benchmark_harness_end_to_end(client: TestClient) -> None:
    """Prove measure -> record -> read-back works (the Phase 0 smoke bench)."""
    run = client.post("/benchmarks/run", json={"suite": "smoke"})
    assert run.status_code == 200
    payload = run.json()
    assert payload["recorded_ids"]
    assert payload["rows"][0]["name"] == "smoke_db_roundtrip"
    assert payload["rows"][0]["value"] >= 0.0

    listed = client.get("/benchmarks")
    assert listed.status_code == 200
    names = {row["name"] for row in listed.json()}
    assert "smoke_db_roundtrip" in names


def test_unknown_benchmark_suite_is_422(client: TestClient) -> None:
    resp = client.post("/benchmarks/run", json={"suite": "does-not-exist"})
    assert resp.status_code == 422
