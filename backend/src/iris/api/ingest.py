"""Ingest control + job endpoints (ARCHITECTURE §8)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from iris.db import jobs as jobs_db
from iris.dependencies import DbDep
from iris.ingest.orchestrator import IngestManager
from iris.schemas import IngestStatus, JobOut, ScanResponse

router = APIRouter(tags=["ingest"])


def _manager(request: Request) -> IngestManager:
    manager: IngestManager = request.app.state.ingest
    return manager


@router.post("/ingest/scan", response_model=ScanResponse)
def scan(request: Request) -> ScanResponse:
    """Start (or resume) ingest over all library roots."""
    manager = _manager(request)
    job_id = manager.start()
    return ScanResponse(job_id=job_id, running=manager.is_running())


@router.get("/ingest/status", response_model=IngestStatus)
def status(request: Request) -> IngestStatus:
    state = _manager(request).status()
    job = state["job"]
    return IngestStatus(running=state["running"], job=JobOut(**job) if job else None)


@router.post("/ingest/cancel")
def cancel(request: Request) -> dict[str, bool]:
    _manager(request).cancel()
    return {"canceled": True}


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(db: DbDep) -> list[JobOut]:
    return [JobOut(**row) for row in jobs_db.list_jobs(db)]


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: DbDep) -> JobOut:
    row = jobs_db.get_job(db, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobOut(**row)
