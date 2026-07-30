"""Library source-root endpoints (ARCHITECTURE §8)."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from iris.db import library as library_db
from iris.dependencies import DbDep
from iris.schemas import RootIn, RootOut

router = APIRouter(prefix="/library", tags=["library"])


@router.get("/roots", response_model=list[RootOut])
def list_roots(db: DbDep) -> list[RootOut]:
    return [RootOut(**row) for row in library_db.list_roots(db)]


@router.post("/roots", response_model=RootOut)
def add_root(body: RootIn, db: DbDep) -> RootOut:
    folder = Path(body.path).expanduser()
    if not folder.is_dir():
        raise HTTPException(status_code=422, detail=f"not a directory: {body.path}")
    root_id = library_db.add_root(db, str(folder.resolve()), time.time())
    row = library_db.get_root(db, root_id)
    assert row is not None
    return RootOut(**row)


@router.delete("/roots/{root_id}")
def remove_root(root_id: int, db: DbDep) -> dict[str, bool]:
    if not library_db.remove_root(db, root_id):
        raise HTTPException(status_code=404, detail="root not found")
    return {"deleted": True}
