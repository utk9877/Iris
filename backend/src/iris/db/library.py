"""DB helpers for library source roots (``roots`` table)."""

from __future__ import annotations

import sqlite3
from typing import Any


def add_root(conn: sqlite3.Connection, path: str, now: float) -> int:
    """Add a source root (idempotent on path). Returns the root id."""
    conn.execute(
        "INSERT INTO roots (path, added_at) VALUES (?, ?) ON CONFLICT(path) DO NOTHING",
        (path, now),
    )
    row = conn.execute("SELECT id FROM roots WHERE path = ?", (path,)).fetchone()
    return int(row[0])


def list_roots(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT id, path, added_at FROM roots ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def get_root(conn: sqlite3.Connection, root_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT id, path, added_at FROM roots WHERE id = ?", (root_id,)).fetchone()
    return dict(row) if row is not None else None


def remove_root(conn: sqlite3.Connection, root_id: int) -> bool:
    """Delete a root. Returns True if a row was removed."""
    cur = conn.execute("DELETE FROM roots WHERE id = ?", (root_id,))
    return cur.rowcount > 0
