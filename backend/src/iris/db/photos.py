"""DB helpers for the ``photos`` table: scan upsert, stage updates, keyset list."""

from __future__ import annotations

import sqlite3
from typing import Any

from iris.ingest.metadata import MetaResult
from iris.ingest.thumbnails import ThumbResult

# A scanned file is "changed" (needs full reprocessing) when its size or mtime differ.
_CHANGED = "(photos.mtime != excluded.mtime OR photos.size_bytes != excluded.size_bytes)"


def upsert_scanned(
    conn: sqlite3.Connection,
    *,
    path: str,
    directory: str,
    filename: str,
    ext: str,
    size_bytes: int,
    mtime: float,
    now: float,
) -> None:
    """Insert a scanned file, or update it and reset derived state if it changed.

    Idempotent: re-scanning an unchanged file is a no-op beyond touching mtime/size,
    so stage markers (and thus completed work) are preserved.
    """
    conn.execute(
        f"""
        INSERT INTO photos (path, dir, filename, ext, size_bytes, mtime, sort_at,
                            missing, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
          size_bytes   = excluded.size_bytes,
          mtime        = excluded.mtime,
          updated_at   = excluded.updated_at,
          missing      = 0,
          content_hash = CASE WHEN {_CHANGED} THEN NULL ELSE photos.content_hash END,
          sort_at      = CASE WHEN {_CHANGED} THEN excluded.sort_at ELSE photos.sort_at END,
          hashed_at    = CASE WHEN {_CHANGED} THEN NULL ELSE photos.hashed_at END,
          exif_at      = CASE WHEN {_CHANGED} THEN NULL ELSE photos.exif_at END,
          thumb_at     = CASE WHEN {_CHANGED} THEN NULL ELSE photos.thumb_at END,
          phash_at     = CASE WHEN {_CHANGED} THEN NULL ELSE photos.phash_at END
        """,
        (path, directory, filename, ext, size_bytes, mtime, mtime, now, now),
    )


def count_pending_thumb(conn: sqlite3.Connection) -> int:
    """Photos still needing to reach the terminal (thumb) stage."""
    row = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE missing = 0 AND thumb_at IS NULL"
    ).fetchone()
    return int(row[0])


def fetch_pending_metadata(conn: sqlite3.Connection, limit: int) -> list[tuple[int, str]]:
    rows = conn.execute(
        "SELECT id, path FROM photos "
        "WHERE missing = 0 AND (hashed_at IS NULL OR exif_at IS NULL) "
        "ORDER BY id LIMIT ?",
        (limit,),
    ).fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


def apply_metadata(conn: sqlite3.Connection, result: MetaResult, now: float) -> None:
    """Persist hash + EXIF for one photo and mark hash/exif stages done."""
    meta = result.meta
    assert result.ok and result.content_hash is not None and meta is not None
    conn.execute(
        """
        UPDATE photos SET
          content_hash = ?, width = ?, height = ?, orientation = ?, taken_at = ?,
          camera_make = ?, camera_model = ?, lens = ?, iso = ?, f_number = ?,
          exposure = ?, focal_length = ?, gps_lat = ?, gps_lon = ?,
          sort_at = COALESCE(?, mtime),
          hashed_at = ?, exif_at = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            result.content_hash,
            meta.width,
            meta.height,
            meta.orientation,
            meta.taken_at,
            meta.camera_make,
            meta.camera_model,
            meta.lens,
            meta.iso,
            meta.f_number,
            meta.exposure,
            meta.focal_length,
            meta.gps_lat,
            meta.gps_lon,
            meta.taken_at,
            now,
            now,
            now,
            result.photo_id,
        ),
    )


def mark_missing(conn: sqlite3.Connection, photo_id: int, now: float) -> None:
    """Flag a photo whose source file could not be read, and skip all stages."""
    conn.execute(
        "UPDATE photos SET missing = 1, hashed_at = ?, exif_at = ?, thumb_at = ?, "
        "updated_at = ? WHERE id = ?",
        (now, now, now, now, photo_id),
    )


def fetch_pending_thumb(conn: sqlite3.Connection, limit: int) -> list[tuple[int, str, str]]:
    rows = conn.execute(
        "SELECT id, path, content_hash FROM photos "
        "WHERE missing = 0 AND content_hash IS NOT NULL AND thumb_at IS NULL "
        "ORDER BY id LIMIT ?",
        (limit,),
    ).fetchall()
    return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]


def apply_thumb(conn: sqlite3.Connection, result: ThumbResult, now: float) -> bool:
    """Persist a thumbnail result. Returns True on success, False if the file failed.

    On failure we still set ``thumb_at`` so a permanently-bad file is not retried
    forever; it simply has no thumbnail (and no phash).
    """
    if result.ok:
        conn.execute(
            "UPDATE photos SET width = ?, height = ?, phash = ?, phash_at = ?, "
            "thumb_at = ?, updated_at = ? WHERE id = ?",
            (result.width, result.height, result.phash, now, now, now, result.photo_id),
        )
        return True
    conn.execute(
        "UPDATE photos SET thumb_at = ?, updated_at = ? WHERE id = ?",
        (now, now, result.photo_id),
    )
    return False


def count_photos(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM photos WHERE missing = 0").fetchone()
    return int(row[0])


_LIST_COLUMNS = "id, filename, content_hash, sort_at, taken_at, width, height, thumb_at"


def list_photos(
    conn: sqlite3.Connection, *, limit: int, cursor: tuple[float, int] | None
) -> list[dict[str, Any]]:
    """Keyset page of photos, newest first, ordered by (sort_at, id) descending."""
    where = "missing = 0"
    params: list[Any] = []
    if cursor is not None:
        sort_at, last_id = cursor
        where += " AND (sort_at < ? OR (sort_at = ? AND id < ?))"
        params += [sort_at, sort_at, last_id]
    params.append(limit)
    rows = conn.execute(
        f"SELECT {_LIST_COLUMNS} FROM photos WHERE {where} ORDER BY sort_at DESC, id DESC LIMIT ?",
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def get_photo(conn: sqlite3.Connection, photo_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
    return dict(row) if row is not None else None


def thumb_content_hash(conn: sqlite3.Connection, photo_id: int) -> str | None:
    """content_hash for a photo that has a rendered thumbnail, else None."""
    row = conn.execute(
        "SELECT content_hash FROM photos WHERE id = ? AND thumb_at IS NOT NULL "
        "AND content_hash IS NOT NULL",
        (photo_id,),
    ).fetchone()
    return str(row[0]) if row is not None else None


# --- Embedding stage (Phase 2) ---


def count_pending_embed(conn: sqlite3.Connection) -> int:
    """Non-missing photos not yet embedded (drives the embed half of job progress)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE missing = 0 AND embed_at IS NULL"
    ).fetchone()
    return int(row[0])


def fetch_pending_embed(conn: sqlite3.Connection, limit: int) -> list[tuple[int, str]]:
    """Photos with a successful thumbnail (phash set) but no embedding yet."""
    rows = conn.execute(
        "SELECT id, content_hash FROM photos "
        "WHERE missing = 0 AND embed_at IS NULL AND phash IS NOT NULL "
        "AND content_hash IS NOT NULL ORDER BY id LIMIT ?",
        (limit,),
    ).fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


def set_embed(conn: sqlite3.Connection, photo_id: int, embed_row: int, now: float) -> None:
    conn.execute(
        "UPDATE photos SET embed_row = ?, embed_at = ?, updated_at = ? WHERE id = ?",
        (embed_row, now, now, photo_id),
    )


def mark_embed_skipped(conn: sqlite3.Connection, photo_id: int, now: float) -> None:
    """Mark a photo embedded-but-vectorless (thumbnail unreadable) so it isn't retried."""
    conn.execute(
        "UPDATE photos SET embed_at = ?, updated_at = ? WHERE id = ?",
        (now, now, photo_id),
    )


def count_pending_faces(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE missing = 0 AND faces_at IS NULL"
    ).fetchone()
    return int(row[0])


def fetch_pending_faces(conn: sqlite3.Connection, limit: int) -> list[tuple[int, str]]:
    """Photos with a rendered thumbnail (phash set) but no face detection yet."""
    rows = conn.execute(
        "SELECT id, path FROM photos "
        "WHERE missing = 0 AND faces_at IS NULL AND phash IS NOT NULL "
        "ORDER BY id LIMIT ?",
        (limit,),
    ).fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


def set_faces_done(conn: sqlite3.Connection, photo_id: int, now: float) -> None:
    conn.execute(
        "UPDATE photos SET faces_at = ?, updated_at = ? WHERE id = ?", (now, now, photo_id)
    )


def photos_by_ids(conn: sqlite3.Connection, ids: list[int]) -> dict[int, dict[str, Any]]:
    """Fetch grid-column rows for a set of ids (for hydrating search results)."""
    result: dict[int, dict[str, Any]] = {}
    for start in range(0, len(ids), 900):  # stay under SQLite's variable cap
        chunk = ids[start : start + 900]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT {_LIST_COLUMNS} FROM photos WHERE id IN ({placeholders})", chunk
        ).fetchall()
        for row in rows:
            result[int(row["id"])] = dict(row)
    return result
