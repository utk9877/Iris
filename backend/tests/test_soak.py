"""Crash-resume soak (ARCHITECTURE §2, Phase 6).

A "kill-storm": start ingest and cancel it at random points many times over, then let a
final run finish. Cancellation mid-stage is the in-process analogue of a crash — stage
markers (`*_at`) are the only durable progress, so a restart must re-process exactly the
unfinished photos and never duplicate or corrupt anything. We assert the DB is fully
processed, consistent, and passes `PRAGMA integrity_check` after the storm.
"""

from __future__ import annotations

import random
import sqlite3
import time
from pathlib import Path

from PIL import Image

from iris.config import Settings
from iris.db import apply_migrations, connect
from iris.db import library as library_db
from iris.db import photos as photos_db
from iris.ingest.orchestrator import IngestManager


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _wait_stopped(manager: IngestManager, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not manager.is_running():
            return
        time.sleep(0.02)
    raise TimeoutError("ingest did not stop in time")


def _seed_library(settings: Settings, tmp_path: Path, n: int = 16) -> None:
    settings.ensure_dirs()
    settings.decode_workers = 2
    conn = _open(settings)
    apply_migrations(conn)
    src = tmp_path / "src"
    src.mkdir()
    rng = random.Random(5)
    for i in range(n):
        color = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        Image.new("RGB", (128, 96), color).save(src / f"img{i:02d}.png")
    library_db.add_root(conn, str(src), time.time())
    conn.close()


def test_kill_storm_leaves_db_consistent(settings: Settings, tmp_path: Path) -> None:
    _seed_library(settings, tmp_path, n=16)
    rng = random.Random(7)

    # Kill-storm: repeatedly start and cancel at random short delays (resume each time).
    for _ in range(6):
        manager = IngestManager(settings)
        manager.start()
        time.sleep(rng.uniform(0.0, 0.25))
        manager.cancel()
        _wait_stopped(manager)

    # Final uninterrupted run to completion.
    final = IngestManager(settings)
    final.start()
    _wait_stopped(final, timeout=90.0)
    assert final.status()["job"]["state"] == "done"

    conn = _open(settings)
    try:
        # Every non-missing photo reached every terminal stage — nothing left pending.
        assert photos_db.count_photos(conn) == 16
        assert photos_db.count_pending_thumb(conn) == 0
        assert photos_db.count_pending_score(conn) == 0
        # No duplicate source rows survived the repeated re-scans.
        dupes = conn.execute(
            "SELECT path, COUNT(*) c FROM photos GROUP BY path HAVING c > 1"
        ).fetchall()
        assert dupes == []
        # Scored photos have real score columns, not NULLs.
        scored = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE scored_at IS NOT NULL "
            "AND (aesthetic IS NULL OR quality IS NULL)"
        ).fetchone()[0]
        assert scored == 0
        # Structural integrity intact after the storm.
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()
