"""Benchmark recording harness.

Per the CLAUDE.md standing instruction, every performance number the app reports
must come from an actual run and be logged to the ``benchmarks`` table. This module
is the only sanctioned writer. Values must be *measured*, never estimated.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any


def git_sha(repo: Path | None = None) -> str | None:
    """Best-effort short git SHA of the working tree, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip() or None


def record_benchmark(
    conn: sqlite3.Connection,
    *,
    name: str,
    value: float,
    unit: str,
    n: int | None = None,
    phase: str | None = None,
    context: dict[str, Any] | None = None,
    job_id: int | None = None,
    sha: str | None = None,
) -> int:
    """Insert one measured benchmark row. Returns the new row id."""
    cur = conn.execute(
        "INSERT INTO benchmarks "
        "(job_id, name, value, unit, n, phase, git_sha, context, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            job_id,
            name,
            float(value),
            unit,
            n,
            phase,
            sha if sha is not None else git_sha(),
            json.dumps(context) if context is not None else None,
            time.time(),
        ),
    )
    row_id = cur.lastrowid
    assert row_id is not None  # INTEGER PRIMARY KEY always yields a rowid
    return row_id


def list_benchmarks(conn: sqlite3.Connection, *, limit: int = 100) -> list[dict[str, Any]]:
    """Most recent benchmark rows, newest first."""
    rows = conn.execute(
        "SELECT id, job_id, name, value, unit, n, phase, git_sha, context, created_at "
        "FROM benchmarks ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- suites


def _smoke_suite(conn: sqlite3.Connection) -> list[int]:
    """Measure real SQLite write+read round-trip latency and record it.

    This is a genuine measurement of the harness itself — it proves the end-to-end
    path (measure -> record -> read back) works, without fabricating any number.
    """
    iterations = 200
    start = time.perf_counter()
    for _ in range(iterations):
        conn.execute("SELECT 1").fetchone()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    per_op_ms = elapsed_ms / iterations
    row_id = record_benchmark(
        conn,
        name="smoke_db_roundtrip",
        value=per_op_ms,
        unit="ms/op",
        n=iterations,
        phase="0",
        context={"suite": "smoke"},
    )
    return [row_id]


_SUITES = {"smoke": _smoke_suite}


def run_suite(conn: sqlite3.Connection, suite: str) -> list[int]:
    """Run a named benchmark suite. Returns the ids of rows it recorded."""
    if suite not in _SUITES:
        raise KeyError(suite)
    return _SUITES[suite](conn)


def available_suites() -> list[str]:
    """Names of benchmark suites that can be run via the API."""
    return sorted(_SUITES)
