"""Tags CRUD + per-photo OCR endpoints (ARCHITECTURE §1/§8)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from iris.db import ocr as ocr_db
from iris.db import photos as photos_db
from iris.db import tags as tags_db
from iris.dependencies import DbDep
from iris.schemas import OcrRegionOut, OcrResponse, PhotoTagRequest, TagOut

router = APIRouter(tags=["tags"])


@router.get("/tags", response_model=list[TagOut])
def list_tags(db: DbDep) -> list[TagOut]:
    return [TagOut(**row) for row in tags_db.list_tags(db)]


@router.get("/photos/{photo_id}/tags", response_model=list[TagOut])
def photo_tags(photo_id: int, db: DbDep) -> list[TagOut]:
    return [
        TagOut(id=t["id"], name=t["name"], kind=t["kind"])
        for t in tags_db.tags_for_photo(db, photo_id)
    ]


@router.post("/photos/{photo_id}/tags", response_model=TagOut)
def add_photo_tag(photo_id: int, body: PhotoTagRequest, db: DbDep) -> TagOut:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="tag name must not be empty")
    if photos_db.get_photo(db, photo_id) is None:
        raise HTTPException(status_code=404, detail="photo not found")
    db.execute("BEGIN")
    try:
        tag = tags_db.get_or_create_tag(db, name)
        tags_db.tag_photo(db, photo_id=photo_id, tag_id=tag["id"])
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return TagOut(id=tag["id"], name=tag["name"], kind=tag["kind"])


@router.delete("/photos/{photo_id}/tags/{tag_id}")
def remove_photo_tag(photo_id: int, tag_id: int, db: DbDep) -> dict[str, bool]:
    tags_db.untag_photo(db, photo_id, tag_id)
    return {"ok": True}


@router.get("/photos/{photo_id}/ocr", response_model=OcrResponse)
def photo_ocr(photo_id: int, db: DbDep) -> OcrResponse:
    data = ocr_db.ocr_for_photo(db, photo_id)
    return OcrResponse(
        photo_id=photo_id,
        text=data["text"],
        regions=[OcrRegionOut(**r) for r in data["regions"]],
    )
