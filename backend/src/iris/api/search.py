"""Semantic + full-text search endpoint (ARCHITECTURE §4/§8).

Semantic (CLIP) ranking is fused with OCR full-text (FTS5/bm25) via reciprocal-rank
fusion so a query like "boarding pass" matches both what a photo *looks like* and the
text printed in it. Text-free libraries fall back transparently to pure semantic ranking.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from iris import search as search_engine
from iris.config import Settings
from iris.db import ocr as ocr_db
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

    candidates = None
    if body.filters is not None:
        candidates = search_engine.build_candidates(
            db,
            folder=body.filters.folder,
            date_from=body.filters.date_from,
            date_to=body.filters.date_to,
        )
        if body.filters.has_text:
            with_text = ocr_db.photos_with_text(db)
            candidates = with_text if candidates is None else (candidates & with_text)

    query_vec = service.embedder().embed_texts([query])[0]
    semantic = search_engine.semantic_search(
        service,
        db,
        query_vec,
        k=limit,
        candidate_ids=candidates,
        brute_max=settings.search_brute_max,
        filtered_max=settings.search_filtered_max,
    )

    # Full-text (OCR) branch — over-fetch so fusion has depth, then apply the filter.
    ocr_ids = ocr_db.search_ocr(db, query, limit * 5)
    if candidates is not None:
        ocr_ids = [pid for pid in ocr_ids if pid in candidates]

    if ocr_ids:
        fused = search_engine.reciprocal_rank_fusion(
            [[hit.photo_id for hit in semantic.hits], ocr_ids]
        )[:limit]
        tier = "fused"
        ranked = fused
    else:
        tier = semantic.tier
        ranked = [(hit.photo_id, hit.score) for hit in semantic.hits]

    rows = photos_db.photos_by_ids(db, [pid for pid, _ in ranked])
    items = [
        SearchItem(
            id=row["id"],
            filename=row["filename"],
            sort_at=row["sort_at"],
            taken_at=row["taken_at"],
            width=row["width"],
            height=row["height"],
            has_thumb=row["thumb_at"] is not None,
            score=score,
        )
        for pid, score in ranked
        if (row := rows.get(pid)) is not None
    ]
    return SearchResponse(tier=tier, items=items)
