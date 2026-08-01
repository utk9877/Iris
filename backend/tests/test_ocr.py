"""OCR: FTS write/search helpers + the ingest OCR stage via a fake engine."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from PIL import Image

from iris.config import Settings
from iris.db import apply_migrations, connect
from iris.db import library as library_db
from iris.db import ocr as ocr_db
from iris.ingest.orchestrator import IngestManager
from iris.ocr.engine import OcrRegion
from iris.ocr.service import OcrService


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _insert_photo(conn: sqlite3.Connection, pid: int) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO photos (id, path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "missing, created_at, updated_at) VALUES (?, ?, '/s', ?, '.jpg', 1, ?, ?, 0, ?, ?)",
        (pid, f"/s/p{pid}.jpg", f"p{pid}.jpg", now, now, now, now),
    )


def test_set_and_search_ocr(settings: Settings) -> None:
    settings.ensure_dirs()
    conn = _open(settings)
    apply_migrations(conn)
    _insert_photo(conn, 1)
    _insert_photo(conn, 2)

    ocr_db.set_ocr(
        conn,
        photo_id=1,
        text="Boarding pass gate 22",
        regions=[(0.1, 0.1, 0.5, 0.1, 0.98, "Boarding pass gate 22")],
        now=time.time(),
    )
    ocr_db.set_ocr(conn, photo_id=2, text="grocery receipt milk", regions=[], now=time.time())

    assert ocr_db.search_ocr(conn, "boarding", 10) == [1]
    assert ocr_db.search_ocr(conn, "milk", 10) == [2]
    assert ocr_db.search_ocr(conn, "nonexistent", 10) == []
    # punctuation / operators in the query must not break the MATCH
    assert ocr_db.search_ocr(conn, "gate #22 (boarding)!", 10) == [1]
    assert ocr_db.photos_with_text(conn) == {1, 2}
    assert ocr_db.ocr_for_photo(conn, 1)["text"] == "Boarding pass gate 22"
    assert len(ocr_db.ocr_for_photo(conn, 1)["regions"]) == 1
    conn.close()


def test_set_ocr_is_idempotent_on_rerun(settings: Settings) -> None:
    settings.ensure_dirs()
    conn = _open(settings)
    apply_migrations(conn)
    _insert_photo(conn, 1)
    ocr_db.set_ocr(conn, photo_id=1, text="old text", regions=[], now=time.time())
    ocr_db.set_ocr(conn, photo_id=1, text="new text", regions=[], now=time.time())
    assert ocr_db.search_ocr(conn, "old", 10) == []
    assert ocr_db.search_ocr(conn, "new", 10) == [1]
    conn.close()


class _FakeOcr:
    """Returns text keyed off the filename so tests can assert on FTS results."""

    def recognize(self, path: str) -> list[OcrRegion]:
        text = "invoice total 42" if "invoice" in path else "vacation beach"
        return [OcrRegion(bx=0.1, by=0.1, bw=0.4, bh=0.1, conf=0.99, text=text)]


def _wait(manager: IngestManager, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not manager.is_running():
            return
        time.sleep(0.1)
    raise TimeoutError


def test_ocr_stage_populates_fts(settings: Settings, tmp_path: Path) -> None:
    settings.ensure_dirs()
    settings.decode_workers = 2
    conn = _open(settings)
    apply_migrations(conn)
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (128, 128), (200, 200, 200)).save(src / "invoice.png")
    Image.new("RGB", (128, 128), (10, 80, 200)).save(src / "holiday.png")
    library_db.add_root(conn, str(src), time.time())
    conn.close()

    ocr = OcrService(settings, engine=_FakeOcr())
    manager = IngestManager(settings, embeddings=None, faces=None, ocr=ocr)
    manager.start()
    _wait(manager)

    conn = _open(settings)
    try:
        assert manager.status()["job"]["state"] == "done"
        assert ocr_db.count_pending_ocr(conn) == 0
        hits = ocr_db.search_ocr(conn, "invoice", 10)
        assert len(hits) == 1
        row = conn.execute("SELECT filename FROM photos WHERE id = ?", (hits[0],)).fetchone()
        assert row[0] == "invoice.png"
    finally:
        conn.close()
