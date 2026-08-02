"""Maintenance endpoints: storage stats + memmap compaction (ARCHITECTURE §3, Phase 6).

Compaction reclaims orphaned vector rows (left by re-embeds / soft-deleted photos) and
renumbers the DB pointers. It mutates the memmaps + DB, so it must not race the ingest
writer — the endpoint returns 409 while an ingest is running.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from iris import cache
from iris.config import Settings
from iris.dependencies import DbDep
from iris.embeddings.service import EmbeddingService
from iris.faces.service import FacesService
from iris.schemas import CacheStat, CompactResponse, CompactStat, StorageStats

router = APIRouter(tags=["maintenance"])


def _stat(used: int, cap_bytes: int | None) -> CacheStat:
    return CacheStat(
        bytes=used, cap_bytes=cap_bytes, over_cap=cap_bytes is not None and used > cap_bytes
    )


@router.get("/maintenance/storage", response_model=StorageStats)
def storage(request: Request) -> StorageStats:
    """Bytes used by each derived-data area vs. its configured cap (ARCHITECTURE §3)."""
    settings: Settings = request.app.state.settings
    gib = 1024**3
    db_bytes = 0
    for suffix in ("", "-wal", "-shm"):
        path = settings.db_path.with_name(settings.db_path.name + suffix)
        if path.exists():
            db_bytes += path.stat().st_size
    return StorageStats(
        thumbs=_stat(cache.tree_size_bytes(settings.thumbs_dir), int(settings.thumb_max_gb * gib)),
        previews=_stat(
            cache.dir_size_bytes(settings.previews_dir), int(settings.preview_cache_gb * gib)
        ),
        embeddings=_stat(cache.tree_size_bytes(settings.embeddings_dir), None),
        database=_stat(db_bytes, None),
    )


@router.post("/maintenance/compact", response_model=CompactResponse)
def compact(request: Request, db: DbDep) -> CompactResponse:
    """Compact the CLIP + face memmaps, reclaiming orphaned rows. 409 if ingest is running."""
    ingest = request.app.state.ingest
    if ingest.is_running():
        raise HTTPException(status_code=409, detail="cannot compact while ingest is running")
    embeddings: EmbeddingService = request.app.state.embeddings
    faces: FacesService = request.app.state.faces
    clip_stats = embeddings.compact()
    face_stats = faces.compact(db)
    return CompactResponse(clip=CompactStat(**clip_stats), faces=CompactStat(**face_stats))
