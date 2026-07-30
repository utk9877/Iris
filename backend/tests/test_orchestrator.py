"""End-to-end ingest orchestrator tests (real thread pool + spawn process pool)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from PIL import Image

from iris.config import Settings
from iris.db import apply_migrations, connect
from iris.db import library as library_db
from iris.db import photos as photos_db
from iris.ingest.orchestrator import IngestManager


def _open_db(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _wait_done(manager: IngestManager, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not manager.is_running():
            return
        time.sleep(0.1)
    raise TimeoutError("ingest did not finish in time")


def _seed_library(settings: Settings, tmp_path: Path) -> Path:
    settings.ensure_dirs()
    settings.decode_workers = 2  # keep the spawn pool small in tests
    conn = _open_db(settings)
    apply_migrations(conn)
    src = tmp_path / "src"
    src.mkdir()
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        Image.new("RGB", (120, 90), color).save(src / f"img{i}.png")
    (src / "bad.png").write_bytes(b"not a real image")  # decodes -> skipped, not fatal
    library_db.add_root(conn, str(src), time.time())
    conn.close()
    return src


def test_full_ingest(settings: Settings, tmp_path: Path) -> None:
    _seed_library(settings, tmp_path)

    manager = IngestManager(settings)
    manager.start()
    _wait_done(manager)

    conn = _open_db(settings)
    try:
        # 3 good + 1 unreadable-but-present file = 4 non-missing rows.
        assert photos_db.count_photos(conn) == 4
        status = manager.status()
        assert status["job"]["state"] == "done"
        assert status["job"]["errored"] == 1  # the bad file

        rows = photos_db.list_photos(conn, limit=10, cursor=None)
        assert len(rows) == 4
        hashed = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE content_hash IS NOT NULL"
        ).fetchone()[0]
        assert hashed == 4
    finally:
        conn.close()

    # 3 thumbnails written to the content-addressed store (the bad file has none).
    thumbs = list(settings.thumbs_dir.rglob("*.webp"))
    assert len(thumbs) == 3


def test_resume_only_processes_pending(settings: Settings, tmp_path: Path) -> None:
    _seed_library(settings, tmp_path)
    manager = IngestManager(settings)
    manager.start()
    _wait_done(manager)

    # Simulate a crash before one photo's thumb stage completed.
    conn = _open_db(settings)
    try:
        conn.execute(
            "UPDATE photos SET thumb_at = NULL, phash = NULL WHERE id = "
            "(SELECT MIN(id) FROM photos WHERE thumb_at IS NOT NULL AND phash IS NOT NULL)"
        )
        assert photos_db.count_pending_thumb(conn) == 1
    finally:
        conn.close()

    manager2 = IngestManager(settings)
    manager2.start()
    _wait_done(manager2)

    conn = _open_db(settings)
    try:
        # The resumed run's total was exactly the 1 pending photo.
        assert manager2.status()["job"]["total"] == 1
        assert photos_db.count_pending_thumb(conn) == 0
    finally:
        conn.close()
