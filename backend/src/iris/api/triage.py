"""Trip-triage endpoints (ARCHITECTURE §7/§8), Phase 5.

``POST /triage`` ranks a scope (a group or a date range) under a preset into a deduped,
diversified shortlist with per-photo score breakdowns and a human reason. ``GET
/triage/presets`` lists the available profiles so the UI can offer a preset switch.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from iris.db import photos as photos_db
from iris.dependencies import DbDep
from iris.schemas import PresetOut, TriageItemOut, TriageRequest, TriageResponse
from iris.triage import PRESETS, TriageService, get_preset

router = APIRouter(tags=["triage"])

_MAX_LIMIT = 500


@router.get("/triage/presets", response_model=list[PresetOut])
def list_presets() -> list[PresetOut]:
    return [
        PresetOut(
            name=p.name,
            description=p.description,
            weights={
                "quality": p.weight_quality,
                "aesthetic": p.weight_aesthetic,
                "representativeness": p.weight_repr,
                "subject": p.weight_subject,
            },
            lam=p.lam,
            invert=p.invert,
        )
        for p in PRESETS.values()
    ]


@router.post("/triage", response_model=TriageResponse)
def triage(body: TriageRequest, request: Request, db: DbDep) -> TriageResponse:
    try:
        preset = get_preset(body.preset)
    except KeyError:
        raise HTTPException(
            status_code=422, detail=f"unknown preset; choose one of {sorted(PRESETS)}"
        ) from None

    service: TriageService = request.app.state.triage
    limit = max(1, min(body.limit, _MAX_LIMIT))
    try:
        result = service.triage(
            db,
            preset=preset,
            group_id=body.group_id,
            date_from=body.date_from,
            date_to=body.date_to,
            limit=limit,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    rows = photos_db.triage_rows(db, [it.photo_id for it in result.items])
    items: list[TriageItemOut] = []
    for it in result.items:
        row: dict[str, Any] | None = rows.get(it.photo_id)
        if row is None:
            continue
        items.append(
            TriageItemOut(
                id=it.photo_id,
                filename=row["filename"],
                sort_at=row["sort_at"],
                taken_at=row["taken_at"],
                width=row["width"],
                height=row["height"],
                has_thumb=row["thumb_at"] is not None,
                score=it.score,
                quality=it.quality,
                aesthetic=it.aesthetic,
                representativeness=it.representativeness,
                subject=it.subject,
                reason=it.reason,
                redundant=it.redundant,
            )
        )
    return TriageResponse(
        preset=result.preset,
        scope_size=result.scope_size,
        considered=result.considered,
        items=items,
    )
