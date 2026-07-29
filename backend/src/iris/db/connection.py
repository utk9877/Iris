"""SQLite connection factory.

Single-writer / many-reader model (ARCHITECTURE §1). WAL is a persistent property
of the database file; the other pragmas are per-connection and applied on every
open. Connections use ``check_same_thread=False`` so a connection may be handed to
a FastAPI threadpool worker, but callers must not share one connection across
threads concurrently without their own serialization.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: Path, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open a configured SQLite connection (creates the file if absent)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,
        isolation_level=None,  # autocommit; we manage transactions explicitly
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")  # safe + fast under WAL
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    return conn
