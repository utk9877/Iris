"""Photo grid + thumbnail endpoints (ARCHITECTURE §8, §9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from iris import cache
from iris.config import Settings
from iris.db import photos as photos_db
from iris.dependencies import DbDep
from iris.schemas import PhotoCount, PhotoOut, PhotoPage
from iris.storage import content_shard_path

router = APIRouter(tags=["photos"])

_MAX_LIMIT = 500


def _encode_cursor(sort_at: float, photo_id: int) -> str:
    return f"{sort_at}:{photo_id}"


def _decode_cursor(raw: str | None) -> tuple[float, int] | None:
    if not raw:
        return None
    try:
        sort_at, photo_id = raw.rsplit(":", 1)
        return float(sort_at), int(photo_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid cursor") from exc


@router.get("/photos", response_model=PhotoPage)
def list_photos(db: DbDep, cursor: str | None = None, limit: int = 100) -> PhotoPage:
    """Keyset page of photos, newest first (ARCHITECTURE §9)."""
    limit = max(1, min(limit, _MAX_LIMIT))
    rows = photos_db.list_photos(db, limit=limit, cursor=_decode_cursor(cursor))
    items = [
        PhotoOut(
            id=row["id"],
            filename=row["filename"],
            sort_at=row["sort_at"],
            taken_at=row["taken_at"],
            width=row["width"],
            height=row["height"],
            has_thumb=row["thumb_at"] is not None,
        )
        for row in rows
    ]
    next_cursor = None
    if len(rows) == limit and rows:
        last = rows[-1]
        next_cursor = _encode_cursor(last["sort_at"], last["id"])
    return PhotoPage(items=items, next_cursor=next_cursor)


@router.get("/photos/count", response_model=PhotoCount)
def count_photos(db: DbDep) -> PhotoCount:
    return PhotoCount(count=photos_db.count_photos(db))


@router.get("/photos/{photo_id}")
def get_photo(photo_id: int, db: DbDep) -> dict[str, Any]:
    row = photos_db.get_photo(db, photo_id)
    if row is None:
        raise HTTPException(status_code=404, detail="photo not found")
    return row


@router.get("/thumb/{photo_id}")
def get_thumb(photo_id: int, request: Request, db: DbDep) -> FileResponse:
    settings: Settings = request.app.state.settings
    digest = photos_db.thumb_content_hash(db, photo_id)
    if digest is None:
        raise HTTPException(status_code=404, detail="no thumbnail")
    path = content_shard_path(settings.thumbs_dir, digest, "webp")
    if not path.exists():
        raise HTTPException(status_code=404, detail="thumbnail file missing")
    return FileResponse(
        path,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/preview/{photo_id}")
def get_preview(photo_id: int, request: Request, db: DbDep) -> FileResponse:
    """Larger (1024 px) preview, rendered on demand from the original and LRU-cached.

    Reads the source read-only (never mutates originals). Falls back to nothing if the
    original is gone or unreadable.
    """
    settings: Settings = request.app.state.settings
    row = photos_db.get_photo(db, photo_id)
    if row is None or row["missing"] or not row["content_hash"]:
        raise HTTPException(status_code=404, detail="no preview")
    src = Path(row["path"])
    if not src.exists():
        raise HTTPException(status_code=404, detail="source file missing")
    try:
        path = cache.ensure_preview(
            settings.previews_dir,
            row["content_hash"],
            src,
            max_edge=settings.preview_max_edge,
            quality=settings.preview_quality,
            cap_bytes=int(settings.preview_cache_gb * 1024**3),
        )
    except Exception as exc:  # unreadable/corrupt source — don't 500
        raise HTTPException(status_code=404, detail="preview render failed") from exc
    return FileResponse(
        path,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=3600"},
    )
