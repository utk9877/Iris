"""Shared pytest fixtures.

Each test gets an isolated app data directory via ``IRIS_DATA_DIR`` so nothing
touches the real ``~/Library/Application Support/Iris``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iris.app import create_app
from iris.config import Settings, get_settings


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "iris-data"))
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app) as test_client:  # triggers lifespan -> migrations
        yield test_client
