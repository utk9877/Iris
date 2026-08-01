"""System endpoints: health, meta, and the benchmark harness (ARCHITECTURE §8)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from iris import __version__, benchmarks
from iris.config import Settings
from iris.db import current_version
from iris.dependencies import DbDep
from iris.schemas import (
    BenchmarkOut,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    HealthResponse,
    MetaResponse,
)

router = APIRouter(tags=["system"])

_COUNT_TABLES = ("photos", "faces", "groups", "tags", "benchmarks")

_INDEX_HTML = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Iris sidecar</title>
<style>
  body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; max-width: 34rem;
         margin: 4rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ margin-bottom: .2rem; }} .v {{ color: #888; }}
  a {{ display: block; padding: .35rem 0; }}
</style></head>
<body>
  <h1>Iris <span class="v">sidecar v{__version__}</span></h1>
  <p>Local-first photo library backend. This is the API host, not the app UI.</p>
  <a href="/health">/health &mdash; liveness</a>
  <a href="/meta">/meta &mdash; schema version &amp; counts</a>
  <a href="/benchmarks">/benchmarks &mdash; recorded runs</a>
  <a href="/docs">/docs &mdash; interactive API docs</a>
</body></html>"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """Landing page so the base URL resolves to something useful (not a 404)."""
    return _INDEX_HTML


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe used by the Tauri shell's readiness handshake."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/meta", response_model=MetaResponse)
def meta(request: Request, db: DbDep) -> MetaResponse:
    """Schema version, row counts, and the active data directory."""
    settings: Settings = request.app.state.settings
    counts = {
        table: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in _COUNT_TABLES
    }
    return MetaResponse(
        app_version=__version__,
        schema_version=current_version(db),
        counts=counts,
        data_dir=str(settings.data_dir),
    )


@router.get("/benchmarks", response_model=list[BenchmarkOut])
def get_benchmarks(db: DbDep, limit: int = 100) -> list[BenchmarkOut]:
    """Recorded benchmark rows, newest first."""
    return [BenchmarkOut(**row) for row in benchmarks.list_benchmarks(db, limit=limit)]


@router.post("/benchmarks/run", response_model=BenchmarkRunResponse)
def run_benchmarks(body: BenchmarkRunRequest, db: DbDep) -> BenchmarkRunResponse:
    """Run a benchmark suite and return the rows it recorded (real measurements)."""
    try:
        recorded = benchmarks.run_suite(db, body.suite)
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"unknown suite {body.suite!r}; available: {benchmarks.available_suites()}",
        ) from exc
    placeholders = ",".join("?" for _ in recorded)
    rows = db.execute(
        "SELECT id, job_id, name, value, unit, n, phase, git_sha, context, created_at "
        f"FROM benchmarks WHERE id IN ({placeholders}) ORDER BY id",
        recorded,
    ).fetchall()
    return BenchmarkRunResponse(
        suite=body.suite,
        recorded_ids=recorded,
        rows=[BenchmarkOut(**dict(row)) for row in rows],
    )
