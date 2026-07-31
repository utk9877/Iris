"""Semantic search endpoint (ARCHITECTURE §4/§8)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from iris import search as search_engine
from iris.config import Settings
from iris.db import photos as photos_db
from iris.dependencies import DbDep
from iris.embeddings.service import EmbeddingService
from iris.schemas import SearchItem, SearchRequest, SearchResponse

router = APIRouter(tags=["search"])

_MAX_LIMIT = 500


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request, db: DbDep) -> SearchResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")
    limit = max(1, min(body.limit, _MAX_LIMIT))

    settings: Settings = request.app.state.settings
    service: EmbeddingService = request.app.state.embeddings

    query_vec = service.embedder().embed_texts([query])[0]
    candidates = None
    if body.filters is not None:
        candidates = search_engine.build_candidates(
            db,
            folder=body.filters.folder,
            date_from=body.filters.date_from,
            date_to=body.filters.date_to,
        )
    result = search_engine.semantic_search(
        service,
        db,
        query_vec,
        k=limit,
        candidate_ids=candidates,
        brute_max=settings.search_brute_max,
        filtered_max=settings.search_filtered_max,
    )

    rows = photos_db.photos_by_ids(db, [hit.photo_id for hit in result.hits])
    items = [
        SearchItem(
            id=row["id"],
            filename=row["filename"],
            sort_at=row["sort_at"],
            taken_at=row["taken_at"],
            width=row["width"],
            height=row["height"],
            has_thumb=row["thumb_at"] is not None,
            score=hit.score,
        )
        for hit in result.hits
        if (row := rows.get(hit.photo_id)) is not None
    ]
    return SearchResponse(tier=result.tier, items=items)
