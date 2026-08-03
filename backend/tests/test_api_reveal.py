"""/photos/{id}/location + /photos/{id}/reveal endpoints and the reveal helper."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iris import reveal
from iris.config import Settings
from iris.db import connect


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _seed(settings: Settings, pid: int, path: str) -> None:
    now = time.time()
    conn = _open(settings)
    conn.execute(
        "INSERT INTO photos (id, path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "missing, created_at, updated_at) VALUES (?, ?, ?, ?, '.jpg', 1, ?, ?, 0, ?, ?)",
        (pid, path, str(Path(path).parent), Path(path).name, now, now, now, now),
    )
    conn.commit()
    conn.close()


def test_location_reports_path_and_existence(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    real = tmp_path / "hero.jpg"
    real.write_bytes(b"jpegbytes")
    _seed(settings, 1, str(real))
    _seed(settings, 2, str(tmp_path / "gone.jpg"))  # never created

    body = client.get("/photos/1/location").json()
    assert body["path"] == str(real)
    assert body["filename"] == "hero.jpg"
    assert body["exists"] is True

    assert client.get("/photos/2/location").json()["exists"] is False
    assert client.get("/photos/999/location").status_code == 404


def test_reveal_invokes_file_manager(
    client: TestClient, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "shot.jpg"
    real.write_bytes(b"x")
    _seed(settings, 1, str(real))

    called: list[Path] = []
    monkeypatch.setattr("iris.api.photos.reveal_in_file_manager", lambda p: called.append(p))

    body = client.post("/photos/1/reveal").json()
    assert body == {"ok": True, "path": str(real)}
    assert called == [real]  # the real path was handed to the file manager


def test_reveal_missing_file_is_404(client: TestClient, settings: Settings, tmp_path: Path) -> None:
    _seed(settings, 1, str(tmp_path / "nope.jpg"))  # not on disk
    assert client.post("/photos/1/reveal").status_code == 404
    assert client.post("/photos/42/reveal").status_code == 404  # unknown photo


def test_reveal_helper_builds_platform_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], bool]] = []

    def fake_run(args: list[str], *, check: bool = False, **kw: object) -> None:
        calls.append((args, check))

    monkeypatch.setattr("iris.reveal.subprocess.run", fake_run)
    monkeypatch.setattr("iris.reveal.sys.platform", "darwin")
    reveal.reveal_in_file_manager(Path("/photos/a.jpg"))
    # macOS reveals with `open -R`, and check=True so a non-zero exit propagates.
    assert calls == [(["open", "-R", "/photos/a.jpg"], True)]
