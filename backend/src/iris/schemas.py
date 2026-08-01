"""Pydantic request/response models for the API (ARCHITECTURE §8)."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


class MetaResponse(BaseModel):
    app_version: str
    schema_version: int
    counts: dict[str, int]
    data_dir: str


class BenchmarkOut(BaseModel):
    id: int
    job_id: int | None
    name: str
    value: float
    unit: str
    n: int | None
    phase: str | None
    git_sha: str | None
    context: str | None
    created_at: float


class BenchmarkRunRequest(BaseModel):
    suite: str = "smoke"


class BenchmarkRunResponse(BaseModel):
    suite: str
    recorded_ids: list[int]
    rows: list[BenchmarkOut]


# --- Library / ingest (ARCHITECTURE §8) ---


class RootIn(BaseModel):
    path: str


class RootOut(BaseModel):
    id: int
    path: str
    added_at: float


class ScanResponse(BaseModel):
    job_id: int
    running: bool


class JobOut(BaseModel):
    id: int
    kind: str
    state: str
    target: str | None
    total: int
    done: int
    errored: int
    checkpoint: str | None
    error_msg: str | None
    started_at: float | None
    updated_at: float | None
    finished_at: float | None


class IngestStatus(BaseModel):
    running: bool
    job: JobOut | None


# --- Photos / grid ---


class PhotoOut(BaseModel):
    id: int
    filename: str
    sort_at: float | None
    taken_at: float | None
    width: int | None
    height: int | None
    has_thumb: bool


class PhotoPage(BaseModel):
    items: list[PhotoOut]
    next_cursor: str | None


class PhotoCount(BaseModel):
    count: int


# --- Search (ARCHITECTURE §4/§8) ---


class SearchFilters(BaseModel):
    folder: str | None = None
    date_from: float | None = None
    date_to: float | None = None
    has_text: bool | None = None  # restrict to photos that contain recognized text


class SearchRequest(BaseModel):
    query: str
    filters: SearchFilters | None = None
    limit: int = 100


class SearchItem(BaseModel):
    id: int
    filename: str
    sort_at: float | None
    taken_at: float | None
    width: int | None
    height: int | None
    has_thumb: bool
    score: float


class SearchResponse(BaseModel):
    tier: str
    items: list[SearchItem]


# --- People / faces (ARCHITECTURE §6/§8) ---


class PersonOut(BaseModel):
    id: int
    label: str | None
    pinned: int
    size: int
    rep_face_id: int | None
    rep_photo_id: int | None


class PersonDetail(BaseModel):
    person: PersonOut
    photos: list[PhotoOut]


class RenameRequest(BaseModel):
    label: str | None = None


class MergeRequest(BaseModel):
    ids: list[int]


class FaceOut(BaseModel):
    id: int
    bx: float
    by: float
    bw: float
    bh: float
    det_score: float
    quality: float | None
    cluster_id: int | None
    assigned: int


# --- Groups (ARCHITECTURE §5/§8) ---


class GroupOut(BaseModel):
    id: int
    kind: str
    key: str | None
    rep_photo_id: int | None
    size: int
    score: float | None
    start_at: float | None
    end_at: float | None


class GroupDetail(BaseModel):
    group: GroupOut
    photos: list[PhotoOut]


# --- Tags + OCR (ARCHITECTURE §1/§8) ---


class TagOut(BaseModel):
    id: int
    name: str
    kind: str
    count: int | None = None


class PhotoTagRequest(BaseModel):
    name: str


class OcrRegionOut(BaseModel):
    bx: float
    by: float
    bw: float
    bh: float
    conf: float
    text: str


class OcrResponse(BaseModel):
    photo_id: int
    text: str
    regions: list[OcrRegionOut]
