"""DB helpers for user/auto tags (ARCHITECTURE §1/§8)."""

from __future__ import annotations

import sqlite3
from typing import Any


def list_tags(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """All tags with a photo count, most-used first."""
    rows = conn.execute(
        "SELECT t.id, t.name, t.kind, COUNT(pt.photo_id) AS count "
        "FROM tags t LEFT JOIN photo_tags pt ON pt.tag_id = t.id "
        "GROUP BY t.id ORDER BY count DESC, t.name"
    ).fetchall()
    return [dict(r) for r in rows]


def get_or_create_tag(conn: sqlite3.Connection, name: str, kind: str = "user") -> dict[str, Any]:
    """Fetch a tag by name (case-preserving, unique), creating it if absent."""
    row = conn.execute("SELECT id, name, kind FROM tags WHERE name = ?", (name,)).fetchone()
    if row is None:
        cur = conn.execute("INSERT INTO tags (name, kind) VALUES (?, ?)", (name, kind))
        return {"id": int(cur.lastrowid or 0), "name": name, "kind": kind}
    return dict(row)


def tag_photo(
    conn: sqlite3.Connection,
    *,
    photo_id: int,
    tag_id: int,
    source: str = "user",
    confidence: float | None = None,
) -> None:
    conn.execute(
        "INSERT INTO photo_tags (photo_id, tag_id, source, confidence) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(photo_id, tag_id) DO UPDATE SET source = excluded.source, "
        "confidence = excluded.confidence",
        (photo_id, tag_id, source, confidence),
    )


def untag_photo(conn: sqlite3.Connection, photo_id: int, tag_id: int) -> None:
    conn.execute("DELETE FROM photo_tags WHERE photo_id = ? AND tag_id = ?", (photo_id, tag_id))


def tags_for_photo(conn: sqlite3.Connection, photo_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT t.id, t.name, t.kind, pt.source, pt.confidence "
        "FROM tags t JOIN photo_tags pt ON pt.tag_id = t.id "
        "WHERE pt.photo_id = ? ORDER BY t.name",
        (photo_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def photos_for_tag(conn: sqlite3.Connection, tag_id: int, limit: int) -> list[int]:
    rows = conn.execute(
        "SELECT photo_id FROM photo_tags WHERE tag_id = ? ORDER BY photo_id DESC LIMIT ?",
        (tag_id, limit),
    ).fetchall()
    return [int(r[0]) for r in rows]
