"""Faces stage integration: orchestrator detects+embeds+clusters via a fake detector."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from iris.config import Settings
from iris.db import apply_migrations, connect
from iris.db import faces as faces_db
from iris.db import library as library_db
from iris.faces.detector import FaceDetector
from iris.faces.service import FacesService
from iris.ingest.orchestrator import IngestManager


class _FakeFace:
    def __init__(self, bbox: tuple[float, ...], kps: np.ndarray, emb: np.ndarray) -> None:
        self.bbox = bbox
        self.kps = kps
        self.det_score = 0.9
        self.normed_embedding = emb


class _FakeApp:
    """One face per image; embedding derived from mean colour -> distinct per photo."""

    def get(self, img: Any) -> list[Any]:
        h, w = img.shape[:2]
        mean = img.reshape(-1, 3).mean(axis=0).astype(np.float32)
        emb = np.zeros(512, dtype=np.float32)
        emb[:3] = mean
        emb[3] = 1.0
        emb /= np.linalg.norm(emb)
        kps = np.array(
            [
                [w * 0.4, h * 0.4],
                [w * 0.6, h * 0.4],
                [w * 0.5, h * 0.5],
                [w * 0.45, h * 0.6],
                [w * 0.55, h * 0.6],
            ],
            dtype=np.float32,
        )
        return [_FakeFace((w * 0.3, h * 0.3, w * 0.7, h * 0.7), kps, emb)]


def _open(settings: Settings) -> sqlite3.Connection:
    return connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)


def _wait(manager: IngestManager, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not manager.is_running():
            return
        time.sleep(0.1)
    raise TimeoutError


def test_faces_stage_detects_embeds_and_clusters(settings: Settings, tmp_path: Path) -> None:
    settings.ensure_dirs()
    settings.decode_workers = 2
    conn = _open(settings)
    apply_migrations(conn)
    src = tmp_path / "src"
    src.mkdir()
    for i, color in enumerate([(230, 20, 20), (20, 230, 20), (20, 20, 230)]):
        Image.new("RGB", (200, 200), color).save(src / f"c{i}.png")
    library_db.add_root(conn, str(src), time.time())
    conn.close()

    detector = FaceDetector(_FakeApp(), min_size=1, min_det_score=0.1)
    faces = FacesService(settings, detector=detector)
    manager = IngestManager(settings, embeddings=None, faces=faces)  # skip embed stage
    manager.start()
    _wait(manager)

    conn = _open(settings)
    try:
        assert manager.status()["job"]["state"] == "done"
        # every photo went through the faces stage, with one face each
        done = conn.execute("SELECT COUNT(*) FROM photos WHERE faces_at IS NOT NULL").fetchone()
        assert done[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 3
        assert faces.store.count == 3

        people = faces_db.list_people(conn)
        assert len(people) == 3  # three distinct colours -> three people
        assert faces_db.count_pending_pool(conn) == 0  # all assigned during update_people
        photo_id = conn.execute("SELECT id FROM photos LIMIT 1").fetchone()[0]
        assert len(faces_db.faces_for_photo(conn, photo_id)) == 1
    finally:
        conn.close()
