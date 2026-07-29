"""FastAPI dependencies."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

from fastapi import Request

from iris.config import Settings
from iris.db import connect


def get_db(request: Request) -> Iterator[sqlite3.Connection]:
    """Yield a fresh SQLite connection per request, closed afterwards.

    A local sidecar sees light request volume, so a per-request connection keeps
    things thread-safe (FastAPI runs sync endpoints in a threadpool) without a
    shared-connection lock. The single-writer discipline for ingest lives elsewhere.
    """
    settings: Settings = request.app.state.settings
    conn = connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    try:
        yield conn
    finally:
        conn.close()
