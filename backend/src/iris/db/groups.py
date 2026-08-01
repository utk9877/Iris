"""DB helpers for photo groupings — events, bursts, near-dups, themes (ARCHITECTURE §5).

Each grouper is idempotent: :func:`replace_groups` clears a whole ``kind`` and rewrites
it, so re-running a grouper after new ingest simply refreshes that layer. ``group_items``
rows cascade-delete with their group.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GroupSpec:
    """One computed group plus its ranked members (rank 0 = representative)."""

    kind: str
    key: str | None = None
    rep_photo_id: int | None = None
    score: float | None = None
    start_at: float | None = None
    end_at: float | None = None
    members: list[tuple[int, float]] = field(default_factory=list)  # (photo_id, rank)


def replace_groups(conn: sqlite3.Connection, kind: str, specs: list[GroupSpec]) -> int:
    """Atomically replace every group of ``kind`` with ``specs``. Returns count written."""
    now = time.time()
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM groups WHERE kind = ?", (kind,))  # cascades group_items
        for spec in specs:
            cur = conn.execute(
                "INSERT INTO groups (kind, key, rep_photo_id, size, score, start_at, end_at, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    kind,
                    spec.key,
                    spec.rep_photo_id,
                    len(spec.members),
                    spec.score,
                    spec.start_at,
                    spec.end_at,
                    now,
                ),
            )
            group_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO group_items (group_id, photo_id, rank) VALUES (?, ?, ?)",
                [(group_id, pid, rank) for pid, rank in spec.members],
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return len(specs)


def count_groups(conn: sqlite3.Connection, kind: str) -> int:
    row = conn.execute("SELECT COUNT(*) FROM groups WHERE kind = ?", (kind,)).fetchone()
    return int(row[0])


def list_groups(
    conn: sqlite3.Connection, *, kind: str, limit: int, offset: int = 0
) -> list[dict[str, Any]]:
    """Groups of one kind, largest first, with the representative photo joined in."""
    rows = conn.execute(
        "SELECT g.id, g.kind, g.key, g.rep_photo_id, g.size, g.score, g.start_at, g.end_at, "
        "p.filename AS rep_filename, p.content_hash AS rep_hash "
        "FROM groups g LEFT JOIN photos p ON p.id = g.rep_photo_id "
        "WHERE g.kind = ? ORDER BY g.size DESC, g.id LIMIT ? OFFSET ?",
        (kind, limit, offset),
    ).fetchall()
    return [dict(r) for r in rows]


def get_group(conn: sqlite3.Connection, group_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT id, kind, key, rep_photo_id, size, score, start_at, end_at "
        "FROM groups WHERE id = ?",
        (group_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def group_photo_ids(conn: sqlite3.Connection, group_id: int) -> list[int]:
    """Member photo ids in rank order (rep first)."""
    rows = conn.execute(
        "SELECT photo_id FROM group_items WHERE group_id = ? ORDER BY rank, photo_id",
        (group_id,),
    ).fetchall()
    return [int(r[0]) for r in rows]


def groups_for_photo(conn: sqlite3.Connection, photo_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT g.id, g.kind, g.key, g.size FROM groups g "
        "JOIN group_items gi ON gi.group_id = g.id WHERE gi.photo_id = ? ORDER BY g.kind",
        (photo_id,),
    ).fetchall()
    return [dict(r) for r in rows]
