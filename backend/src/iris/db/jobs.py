"""DB helpers for the ``jobs`` table (ingest progress tracking, ARCHITECTURE §2)."""

from __future__ import annotations

import sqlite3
from typing import Any

_COLUMNS = (
    "id, kind, state, target, total, done, errored, checkpoint, error_msg, "
    "started_at, updated_at, finished_at"
)


def create_job(
    conn: sqlite3.Connection, *, kind: str, target: str | None, total: int, now: float
) -> int:
    cur = conn.execute(
        "INSERT INTO jobs (kind, state, target, total, done, errored, started_at, updated_at) "
        "VALUES (?, 'running', ?, ?, 0, 0, ?, ?)",
        (kind, target, total, now, now),
    )
    job_id = cur.lastrowid
    assert job_id is not None
    return job_id


def update_job(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    now: float,
    total: int | None = None,
    done: int | None = None,
    errored: int | None = None,
    state: str | None = None,
    error_msg: str | None = None,
    finished: bool = False,
) -> None:
    sets: list[str] = ["updated_at = ?"]
    params: list[Any] = [now]
    for column, value in (
        ("total", total),
        ("done", done),
        ("errored", errored),
        ("state", state),
        ("error_msg", error_msg),
    ):
        if value is not None:
            sets.append(f"{column} = ?")
            params.append(value)
    if finished:
        sets.append("finished_at = ?")
        params.append(now)
    params.append(job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)


def get_job(conn: sqlite3.Connection, job_id: int) -> dict[str, Any] | None:
    row = conn.execute(f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row is not None else None


def list_jobs(conn: sqlite3.Connection, *, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(row) for row in rows]


def fail_running_jobs(conn: sqlite3.Connection, *, kind: str, now: float) -> None:
    """Mark any lingering 'running' jobs of a kind as errored (crash cleanup)."""
    conn.execute(
        "UPDATE jobs SET state = 'error', error_msg = 'interrupted', "
        "updated_at = ?, finished_at = ? WHERE kind = ? AND state = 'running'",
        (now, now, kind),
    )
