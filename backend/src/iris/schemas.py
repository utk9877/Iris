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
