"""Semantic + full-text + date/location search endpoint (ARCHITECTURE §4/§8).

Semantic (CLIP) ranking fuses with OCR full-text (FTS5/bm25) via reciprocal-rank fusion.
Structured filters narrow the candidate set first: **date** (explicit, or parsed from
natural language in the query itself — "beach 2024", "last summer"), **location** (a place
name resolved offline to coordinates, or "near this photo"), folder, and has_text. When the
query is *only* a date/place ("photos from 2024"), it becomes a filter-only browse ordered
by recency — no semantic ranking needed.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from iris import search as search_engine
from iris.config import Settings
from iris.db import ocr as ocr_db
from iris.db import photos as photos_db
from iris.dependencies import DbDep
from iris.embeddings.service import EmbeddingService
from iris.geo.service import GeoService
from iris.query_parse import parse_dates
from iris.schemas import AppliedFilters, SearchItem, SearchRequest, SearchResponse

router = APIRouter(tags=["search"])

_MAX_LIMIT = 500


def _item(row: dict[str, Any], score: float) -> SearchItem:
    return SearchItem(
        id=row["id"],
        filename=row["filename"],
        sort_at=row["sort_at"],
        taken_at=row["taken_at"],
        width=row["width"],
        height=row["height"],
        has_thumb=row["thumb_at"] is not None,
        score=score,
    )


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request, db: DbDep) -> SearchResponse:
    limit = max(1, min(body.limit, _MAX_LIMIT))
    settings: Settings = request.app.state.settings
    service: EmbeddingService = request.app.state.embeddings
    geo: GeoService = request.app.state.geo
    filters = body.filters

    # 1. Natural-language date parsing from the query; explicit filter dates win.
    parsed = parse_dates(body.query)
    text_query = parsed.cleaned_query.strip()
    date_from = filters.date_from if filters and filters.date_from is not None else parsed.date_from
    date_to = filters.date_to if filters and filters.date_to is not None else parsed.date_to
    date_label = None if (filters and filters.date_from is not None) else parsed.label

    # 2. Resolve a location filter (place name or "near this photo") to center + radius.
    gps_center: tuple[float, float] | None = None
    radius_km = (
        filters.radius_km if filters and filters.radius_km else settings.geo_default_radius_km
    )
    place_label: str | None = None
    place_error: str | None = None
    if filters and filters.place:
        place = geo.resolve_place(filters.place)
        if place is None:
            place_error = f"couldn't locate {filters.place!r} (is the geo data installed?)"
        else:
            gps_center, place_label = (place.lat, place.lon), place.label
    elif filters and filters.near_photo_id is not None:
        gps_center = search_engine.photo_gps(db, filters.near_photo_id)
        if gps_center is None:
            place_error = "that photo has no GPS location"
        else:
            place_label = "near selected photo"

    has_filter = bool(
        (filters and (filters.folder or filters.has_text))
        or date_from is not None
        or date_to is not None
        or gps_center is not None
    )
    candidates = None
    if has_filter:
        candidates = search_engine.build_candidates(
            db,
            folder=filters.folder if filters else None,
            date_from=date_from,
            date_to=date_to,
            gps_center=gps_center,
            gps_radius_km=radius_km,
        )
        if filters and filters.has_text:
            with_text = ocr_db.photos_with_text(db)
            candidates = with_text if candidates is None else (candidates & with_text)

    applied = AppliedFilters(
        date_label=date_label,
        date_from=date_from,
        date_to=date_to,
        place_label=place_label,
        radius_km=radius_km if gps_center is not None else None,
        text_query=text_query or None,
    )

    # 3. Filter-only browse: no words left to rank on -> newest matching photos.
    if not text_query:
        if candidates is None:
            raise HTTPException(status_code=422, detail="empty query and no filters")
        rows = photos_db.photos_by_ids(db, list(candidates))
        ordered = sorted(rows.values(), key=lambda r: r["sort_at"] or 0.0, reverse=True)[:limit]
        return SearchResponse(
            tier="filter",
            items=[_item(r, r["sort_at"] or 0.0) for r in ordered],
            applied=applied,
            place_error=place_error,
        )

    # 4. Semantic + OCR fusion over the (optional) candidate set.
    query_vec = service.embedder().embed_texts([text_query])[0]
    semantic = search_engine.semantic_search(
        service,
        db,
        query_vec,
        k=limit,
        candidate_ids=candidates,
        brute_max=settings.search_brute_max,
        filtered_max=settings.search_filtered_max,
    )
    ocr_ids = ocr_db.search_ocr(db, text_query, limit * 5)
    if candidates is not None:
        ocr_ids = [pid for pid in ocr_ids if pid in candidates]

    if ocr_ids:
        ranked = search_engine.reciprocal_rank_fusion(
            [[hit.photo_id for hit in semantic.hits], ocr_ids]
        )[:limit]
        tier = "fused"
    else:
        tier = semantic.tier
        ranked = [(hit.photo_id, hit.score) for hit in semantic.hits]

    rows = photos_db.photos_by_ids(db, [pid for pid, _ in ranked])
    items = [_item(row, score) for pid, score in ranked if (row := rows.get(pid)) is not None]
    return SearchResponse(tier=tier, items=items, applied=applied, place_error=place_error)
