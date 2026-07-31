"""Ingest orchestrator (ARCHITECTURE §2).

A single background thread owns the one SQLite *writer* connection and drives the
pipeline: scan -> metadata (hash+EXIF, thread pool) -> decode+thumb+phash (spawn
process pool). Stages run sequentially (thumb needs the hash), parallel within.
Crash-resume is implicit: every stage records a per-photo ``*_at`` marker, so a
restart re-processes only rows whose marker is still NULL.
"""

from __future__ import annotations

import logging
import multiprocessing
import sqlite3
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from iris.config import Settings
from iris.db import connect
from iris.db import faces as faces_db
from iris.db import jobs as jobs_db
from iris.db import library as library_db
from iris.db import photos as photos_db
from iris.embeddings.service import EmbeddingService
from iris.faces.detector import load_image_bgr
from iris.faces.service import FacesService
from iris.ingest.metadata import extract_metadata
from iris.ingest.scanner import iter_image_files
from iris.ingest.thumbnails import render_thumbnail
from iris.storage import content_shard_path

logger = logging.getLogger("iris.ingest")

_SCAN_COMMIT_EVERY = 1000
_METADATA_BATCH = 256
_THUMB_BATCH = 128


class IngestManager:
    """Owns the ingest background thread and exposes start/cancel/status."""

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService | None = None,
        faces: FacesService | None = None,
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._faces = faces
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._job_id: int | None = None
        self._done = 0
        self._errored = 0

    # ------------------------------------------------------------------ public
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> int:
        """Start (or resume) ingest. Idempotent while a run is in flight."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                assert self._job_id is not None
                return self._job_id
            self._cancel.clear()
            self._done = 0
            self._errored = 0
            now = time.time()
            with self._writer() as conn:
                jobs_db.fail_running_jobs(conn, kind="ingest", now=now)
                job_id = jobs_db.create_job(conn, kind="ingest", target="library", total=0, now=now)
            self._job_id = job_id
            self._thread = threading.Thread(
                target=self._run, args=(job_id,), name="iris-ingest", daemon=True
            )
            self._thread.start()
            return job_id

    def cancel(self) -> None:
        self._cancel.set()

    def status(self) -> dict[str, Any]:
        job = None
        if self._job_id is not None:
            with self._reader() as conn:
                job = jobs_db.get_job(conn, self._job_id)
        return {"running": self.is_running(), "job": job}

    # ----------------------------------------------------------------- helpers
    def _writer(self) -> Any:
        return _ConnCtx(self._settings)

    _reader = _writer  # same config; WAL permits concurrent readers

    # --------------------------------------------------------------- pipeline
    def _run(self, job_id: int) -> None:
        conn = connect(
            self._settings.db_path, busy_timeout_ms=self._settings.sqlite_busy_timeout_ms
        )
        try:
            self._scan(conn)
            total = photos_db.count_pending_thumb(conn)
            if self._embeddings is not None:
                total += photos_db.count_pending_embed(conn)
            if self._faces is not None:
                total += photos_db.count_pending_faces(conn)
            jobs_db.update_job(conn, job_id, now=time.time(), total=total)

            self._run_metadata(conn)
            if not self._cancel.is_set():
                self._run_thumbs(conn, job_id)
            if self._embeddings is not None and not self._cancel.is_set():
                self._run_embed(conn, job_id, self._embeddings)
            if self._faces is not None and not self._cancel.is_set():
                self._run_faces(conn, job_id, self._faces)

            state = "canceled" if self._cancel.is_set() else "done"
            if state == "done":
                self._done = total  # clamp to 100% (thumb failures leave a small gap)
            jobs_db.update_job(
                conn,
                job_id,
                now=time.time(),
                state=state,
                done=self._done,
                errored=self._errored,
                finished=True,
            )
        except Exception as exc:  # pipeline-level failure
            logger.exception("ingest failed")
            jobs_db.update_job(
                conn,
                job_id,
                now=time.time(),
                state="error",
                error_msg=str(exc),
                finished=True,
            )
        finally:
            conn.close()

    def _scan(self, conn: sqlite3.Connection) -> None:
        now = time.time()
        pending = 0
        conn.execute("BEGIN")
        try:
            for root in library_db.list_roots(conn):
                base = Path(root["path"])
                if not base.exists():
                    continue
                for path in iter_image_files(base):
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    photos_db.upsert_scanned(
                        conn,
                        path=str(path),
                        directory=str(path.parent),
                        filename=path.name,
                        ext=path.suffix.lower(),
                        size_bytes=stat.st_size,
                        mtime=stat.st_mtime,
                        now=now,
                    )
                    pending += 1
                    if pending % _SCAN_COMMIT_EVERY == 0:
                        conn.execute("COMMIT")
                        conn.execute("BEGIN")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _run_metadata(self, conn: sqlite3.Connection) -> None:
        with ThreadPoolExecutor(max_workers=self._settings.metadata_workers) as pool:
            while not self._cancel.is_set():
                batch = photos_db.fetch_pending_metadata(conn, _METADATA_BATCH)
                if not batch:
                    break
                futures = [pool.submit(extract_metadata, pid, path) for pid, path in batch]
                conn.execute("BEGIN")
                try:
                    for future in as_completed(futures):
                        result = future.result()
                        if result.ok:
                            photos_db.apply_metadata(conn, result, time.time())
                        else:
                            photos_db.mark_missing(conn, result.photo_id, time.time())
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise

    def _run_thumbs(self, conn: sqlite3.Connection, job_id: int) -> None:
        ctx = multiprocessing.get_context("spawn")
        workers = self._settings.resolved_decode_workers()
        thumbs_dir = str(self._settings.thumbs_dir)
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        try:
            while not self._cancel.is_set():
                batch = photos_db.fetch_pending_thumb(conn, _THUMB_BATCH)
                if not batch:
                    break
                futures = [
                    pool.submit(render_thumbnail, pid, path, chash, thumbs_dir)
                    for pid, path, chash in batch
                ]
                try:
                    self._collect_thumbs(conn, futures)
                except BrokenProcessPool:
                    # A native decoder crash killed a worker. Skip this batch so we
                    # make progress, then start a fresh pool. (Per-file isolation is
                    # a future refinement — see ARCHITECTURE §10 risk #1.)
                    logger.warning("decode worker crashed; skipping batch of %d", len(batch))
                    self._skip_thumbs(conn, [pid for pid, _, _ in batch])
                    pool.shutdown(wait=False, cancel_futures=True)
                    pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
                jobs_db.update_job(
                    conn, job_id, now=time.time(), done=self._done, errored=self._errored
                )
        finally:
            pool.shutdown(wait=True)

    def _collect_thumbs(self, conn: sqlite3.Connection, futures: list[Any]) -> None:
        conn.execute("BEGIN")
        try:
            for future in as_completed(futures):
                result = future.result()
                if not photos_db.apply_thumb(conn, result, time.time()):
                    self._errored += 1
                self._done += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _skip_thumbs(self, conn: sqlite3.Connection, photo_ids: list[int]) -> None:
        now = time.time()
        conn.execute("BEGIN")
        try:
            for photo_id in photo_ids:
                conn.execute(
                    "UPDATE photos SET thumb_at = ?, updated_at = ? WHERE id = ? "
                    "AND thumb_at IS NULL",
                    (now, now, photo_id),
                )
                self._done += 1
                self._errored += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _run_embed(
        self, conn: sqlite3.Connection, job_id: int, embeddings: EmbeddingService
    ) -> None:
        """Embed thumbnails via CLIP, append to the memmap, and index them.

        Reads the already-rendered thumbnail (not the original) to avoid a second
        decode — a Phase 2 simplification vs. the shared-memory tensor hand-off in
        ARCHITECTURE §2. Runs one ONNX session batched-serial in this thread.
        """
        embedder = embeddings.embedder()  # lazy load (may download on first use)
        embeddings.ensure_index()
        thumbs_dir = self._settings.thumbs_dir
        batch_size = self._settings.embed_batch

        while not self._cancel.is_set():
            batch = photos_db.fetch_pending_embed(conn, batch_size)
            if not batch:
                break
            images: list[Image.Image] = []
            ids: list[int] = []
            skipped: list[int] = []
            for photo_id, digest in batch:
                path = content_shard_path(thumbs_dir, digest, "webp")
                try:
                    images.append(Image.open(path).convert("RGB"))
                    ids.append(photo_id)
                except Exception:
                    skipped.append(photo_id)

            now = time.time()
            if images:
                vectors = embedder.embed_images(images)
                start_row = embeddings.store.append(vectors)
                conn.execute("BEGIN")
                try:
                    for offset, photo_id in enumerate(ids):
                        photos_db.set_embed(conn, photo_id, start_row + offset, now)
                    for photo_id in skipped:
                        photos_db.mark_embed_skipped(conn, photo_id, now)
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                embeddings.add_to_index(vectors, np.array(ids, dtype=np.int64))
                self._done += len(ids)
            if skipped and not images:
                conn.execute("BEGIN")
                for photo_id in skipped:
                    photos_db.mark_embed_skipped(conn, photo_id, now)
                conn.execute("COMMIT")
            self._done += len(skipped)
            self._errored += len(skipped)
            jobs_db.update_job(
                conn, job_id, now=time.time(), done=self._done, errored=self._errored
            )

        embeddings.persist_index()

    def _run_faces(self, conn: sqlite3.Connection, job_id: int, faces: FacesService) -> None:
        """Detect + embed faces per photo, write faces + memmap, then update people (§6)."""
        try:
            detector = faces.detector()  # lazy load (may download / need the faces extra)
        except Exception:
            # A missing/broken face model must not fail the whole ingest — skip the
            # faces stage (photos keep faces_at NULL and are retried on a later scan).
            logger.warning("face detector unavailable; skipping faces stage", exc_info=True)
            return
        max_edge = self._settings.face_max_edge
        batch_size = self._settings.face_batch

        while not self._cancel.is_set():
            batch = photos_db.fetch_pending_faces(conn, batch_size)
            if not batch:
                break
            conn.execute("BEGIN")
            try:
                for photo_id, path in batch:
                    try:
                        image = load_image_bgr(path, max_edge)
                        detected = detector.detect(image)
                    except Exception as exc:
                        logger.warning("face detect failed for %s: %s", path, exc)
                        detected = []
                    for face in detected:
                        embed_row = faces.store.append(face.embedding.reshape(1, -1))
                        faces_db.insert_face(
                            conn,
                            photo_id=photo_id,
                            bbox=face.bbox,
                            det_score=face.det_score,
                            landmarks=face.landmarks,
                            quality=face.quality,
                            embed_row=embed_row,
                            now=time.time(),
                        )
                    photos_db.set_faces_done(conn, photo_id, time.time())
                    self._done += 1
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            jobs_db.update_job(
                conn, job_id, now=time.time(), done=self._done, errored=self._errored
            )

        if not self._cancel.is_set():
            faces.update_people(conn)


class _ConnCtx:
    """Small context manager yielding a configured connection, closed on exit."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self._conn = connect(
            self._settings.db_path, busy_timeout_ms=self._settings.sqlite_busy_timeout_ms
        )
        return self._conn

    def __exit__(self, *exc: object) -> None:
        if self._conn is not None:
            self._conn.close()
