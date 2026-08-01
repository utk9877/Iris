"""DB helpers for OCR full-text (FTS5) and per-region boxes (ARCHITECTURE §1/§4).

``ocr`` is a small FTS5 table (one row per photo, concatenated text) queried with
``MATCH`` + ``bm25`` ranking; ``ocr_regions`` keeps per-box detail for highlighting.
Writers do not open their own transaction — the ingest stage batches many photos per
commit, mirroring the faces stage.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

# FTS5 treats many characters as query operators; for user text we tokenize into bare
# words and OR them as quoted terms, so a query like "invoice #7 (paid)" can't break MATCH.
_WORD = re.compile(r"\w+", re.UNICODE)


def build_match(query: str) -> str | None:
    """Turn free text into a safe FTS5 MATCH expression, or None if it has no words."""
    words = _WORD.findall(query)
    if not words:
        return None
    return " OR ".join(f'"{w}"' for w in words)


def set_ocr(
    conn: sqlite3.Connection,
    *,
    photo_id: int,
    text: str,
    regions: list[tuple[float, float, float, float, float, str]],
    now: float,
) -> None:
    """Replace a photo's OCR text + regions and mark the stage done.

    ``regions`` items are ``(bx, by, bw, bh, conf, text)`` with a normalized bbox.
    Re-running (e.g. after the source changed) is safe: prior rows are cleared first.
    """
    conn.execute("DELETE FROM ocr WHERE photo_id = ?", (photo_id,))
    conn.execute("DELETE FROM ocr_regions WHERE photo_id = ?", (photo_id,))
    if text.strip():
        conn.execute("INSERT INTO ocr (text, photo_id) VALUES (?, ?)", (text, photo_id))
    conn.executemany(
        "INSERT INTO ocr_regions (photo_id, bx, by, bw, bh, conf, text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(photo_id, bx, by, bw, bh, conf, rtext) for bx, by, bw, bh, conf, rtext in regions],
    )
    conn.execute("UPDATE photos SET ocr_at = ?, updated_at = ? WHERE id = ?", (now, now, photo_id))


def count_pending_ocr(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE missing = 0 AND ocr_at IS NULL"
    ).fetchone()
    return int(row[0])


def fetch_pending_ocr(conn: sqlite3.Connection, limit: int) -> list[tuple[int, str]]:
    """Photos with a rendered thumbnail (phash set) but no OCR yet."""
    rows = conn.execute(
        "SELECT id, path FROM photos "
        "WHERE missing = 0 AND ocr_at IS NULL AND phash IS NOT NULL "
        "ORDER BY id LIMIT ?",
        (limit,),
    ).fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


def search_ocr(conn: sqlite3.Connection, query: str, limit: int) -> list[int]:
    """Photo ids whose OCR text matches ``query``, ranked by bm25 (best first)."""
    match = build_match(query)
    if match is None:
        return []
    rows = conn.execute(
        "SELECT photo_id FROM ocr WHERE ocr MATCH ? ORDER BY bm25(ocr) LIMIT ?",
        (match, limit),
    ).fetchall()
    return [int(r[0]) for r in rows]


def photos_with_text(conn: sqlite3.Connection) -> set[int]:
    """Every photo id that has any OCR text — backs the ``has_text`` search filter."""
    rows = conn.execute("SELECT DISTINCT photo_id FROM ocr").fetchall()
    return {int(r[0]) for r in rows}


def ocr_for_photo(conn: sqlite3.Connection, photo_id: int) -> dict[str, Any]:
    """Concatenated text + region boxes for one photo (empty lists if none)."""
    text_row = conn.execute("SELECT text FROM ocr WHERE photo_id = ?", (photo_id,)).fetchone()
    regions = conn.execute(
        "SELECT bx, by, bw, bh, conf, text FROM ocr_regions WHERE photo_id = ? ORDER BY id",
        (photo_id,),
    ).fetchall()
    return {
        "text": str(text_row[0]) if text_row is not None else "",
        "regions": [dict(r) for r in regions],
    }
