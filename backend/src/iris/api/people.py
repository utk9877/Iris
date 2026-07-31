"""People (face-cluster) + faces endpoints (ARCHITECTURE §6/§8)."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from iris.db import faces as faces_db
from iris.db import photos as photos_db
from iris.dependencies import DbDep
from iris.faces.service import FacesService
from iris.schemas import (
    FaceOut,
    MergeRequest,
    PersonDetail,
    PersonOut,
    PhotoOut,
    RenameRequest,
)

router = APIRouter(tags=["people"])


def _faces(request: Request) -> FacesService:
    service: FacesService = request.app.state.faces
    return service


@router.get("/people", response_model=list[PersonOut])
def list_people(db: DbDep) -> list[PersonOut]:
    return [PersonOut(**row) for row in faces_db.list_people(db)]


@router.get("/people/{cluster_id}", response_model=PersonDetail)
def get_person(cluster_id: int, db: DbDep) -> PersonDetail:
    person = faces_db.get_person(db, cluster_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")
    photo_ids = faces_db.photos_for_person(db, cluster_id, limit=500)
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
    return PersonDetail(person=PersonOut(**person), photos=photos)


@router.patch("/people/{cluster_id}", response_model=PersonOut)
def rename_person(cluster_id: int, body: RenameRequest, db: DbDep) -> PersonOut:
    if faces_db.get_person(db, cluster_id) is None:
        raise HTTPException(status_code=404, detail="person not found")
    label = body.label.strip() if body.label else None
    faces_db.set_label(db, cluster_id, label or None, time.time())
    updated = faces_db.get_person(db, cluster_id)
    assert updated is not None
    return PersonOut(**updated)


@router.post("/people/merge", response_model=PersonOut)
def merge_people(body: MergeRequest, request: Request, db: DbDep) -> PersonOut:
    if len(body.ids) < 2:
        raise HTTPException(status_code=422, detail="need at least two people to merge")
    target, others = body.ids[0], body.ids[1:]
    _faces(request).merge_clusters(db, target, others)
    person = faces_db.get_person(db, target)
    assert person is not None
    return PersonOut(**person)


@router.post("/people/{cluster_id}/split")
def split_person(cluster_id: int, request: Request, db: DbDep) -> dict[str, int]:
    if faces_db.get_person(db, cluster_id) is None:
        raise HTTPException(status_code=404, detail="person not found")
    return {"clusters": _faces(request).split_cluster(db, cluster_id)}


@router.post("/faces/recluster")
def recluster(request: Request, db: DbDep) -> dict[str, int]:
    _faces(request).update_people(db, force=True)
    return {"people": len(faces_db.list_people(db))}


@router.get("/photos/{photo_id}/faces", response_model=list[FaceOut])
def photo_faces(photo_id: int, db: DbDep) -> list[FaceOut]:
    return [FaceOut(**row) for row in faces_db.faces_for_photo(db, photo_id)]
