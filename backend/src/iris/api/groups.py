"""Groups (events / bursts / near-dups / themes) endpoints (ARCHITECTURE §5/§8)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from iris.db import groups as groups_db
from iris.db import photos as photos_db
from iris.dependencies import DbDep
from iris.schemas import GroupDetail, GroupOut, PhotoOut

router = APIRouter(tags=["groups"])

_KINDS = {"event", "burst", "near_dup", "semantic"}
_MAX_LIMIT = 500


@router.get("/groups", response_model=list[GroupOut])
def list_groups(db: DbDep, kind: str, limit: int = 100, offset: int = 0) -> list[GroupOut]:
    if kind not in _KINDS:
        raise HTTPException(status_code=422, detail=f"kind must be one of {sorted(_KINDS)}")
    limit = max(1, min(limit, _MAX_LIMIT))
    rows = groups_db.list_groups(db, kind=kind, limit=limit, offset=max(0, offset))
    return [
        GroupOut(
            id=r["id"],
            kind=r["kind"],
            key=r["key"],
            rep_photo_id=r["rep_photo_id"],
            size=r["size"],
            score=r["score"],
            start_at=r["start_at"],
            end_at=r["end_at"],
        )
        for r in rows
    ]


@router.get("/groups/{group_id}", response_model=GroupDetail)
def get_group(group_id: int, db: DbDep) -> GroupDetail:
    group = groups_db.get_group(db, group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="group not found")
    photo_ids = groups_db.group_photo_ids(db, group_id)
    rows = photos_db.photos_by_ids(db, photo_ids)
    photos = [
        PhotoOut(
            id=row["id"],
            filename=row["filename"],
            sort_at=row["sort_at"],
            taken_at=row["taken_at"],
            width=row["width"],
            height=row["height"],
            has_thumb=row["thumb_at"] is not None,
        )
        for pid in photo_ids
        if (row := rows.get(pid)) is not None
    ]
    return GroupDetail(group=GroupOut(**group), photos=photos)
