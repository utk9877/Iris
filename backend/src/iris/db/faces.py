"""DB helpers for the ``faces`` and ``clusters`` (person) tables (ARCHITECTURE §6)."""

from __future__ import annotations

import sqlite3
from typing import Any


def insert_face(
    conn: sqlite3.Connection,
    *,
    photo_id: int,
    bbox: tuple[float, float, float, float],
    det_score: float,
    landmarks: bytes | None,
    quality: float,
    embed_row: int,
    now: float,
) -> int:
    bx, by, bw, bh = bbox
    cur = conn.execute(
        "INSERT INTO faces (photo_id, bx, by, bw, bh, det_score, landmarks, quality, "
        "embed_row, assigned, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
        (photo_id, bx, by, bw, bh, det_score, landmarks, quality, embed_row, now),
    )
    assert cur.lastrowid is not None
    return cur.lastrowid


def faces_for_photo(conn: sqlite3.Connection, photo_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, bx, by, bw, bh, det_score, quality, cluster_id, assigned "
        "FROM faces WHERE photo_id = ? ORDER BY id",
        (photo_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def faces_with_person(
    conn: sqlite3.Connection, photo_ids: list[int]
) -> dict[int, list[tuple[float, bool]]]:
    """Map ``photo_id -> [(face_quality, is_named_person), ...]`` for the given photos.

    Feeds the triage *subject* score (ARCHITECTURE §7): face presence weighted by face
    quality, with a bonus when the face belongs to a labeled (named) person cluster.
    """
    result: dict[int, list[tuple[float, bool]]] = {}
    for start in range(0, len(photo_ids), 900):
        chunk = photo_ids[start : start + 900]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            "SELECT f.photo_id, f.quality, c.label FROM faces f "
            "LEFT JOIN clusters c ON c.id = f.cluster_id "
            f"WHERE f.photo_id IN ({placeholders})",
            chunk,
        ).fetchall()
        for row in rows:
            named = row["label"] is not None and str(row["label"]).strip() != ""
            result.setdefault(int(row["photo_id"]), []).append(
                (float(row["quality"] or 0.0), named)
            )
    return result


def pending_faces(conn: sqlite3.Connection) -> list[tuple[int, int, float]]:
    """(face_id, embed_row, quality) for faces not yet assigned to a person."""
    rows = conn.execute(
        "SELECT id, embed_row, quality FROM faces "
        "WHERE assigned = 0 AND embed_row IS NOT NULL ORDER BY id"
    ).fetchall()
    return [(int(r[0]), int(r[1]), float(r[2] or 0.0)) for r in rows]


def count_pending_pool(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM faces WHERE assigned = 0").fetchone()
    return int(row[0])


def cluster_face_rows(conn: sqlite3.Connection, cluster_id: int) -> list[tuple[int, int, float]]:
    rows = conn.execute(
        "SELECT id, embed_row, quality FROM faces WHERE cluster_id = ? AND embed_row IS NOT NULL",
        (cluster_id,),
    ).fetchall()
    return [(int(r[0]), int(r[1]), float(r[2] or 0.0)) for r in rows]


def assign_face(conn: sqlite3.Connection, face_id: int, cluster_id: int) -> None:
    conn.execute(
        "UPDATE faces SET cluster_id = ?, assigned = 1 WHERE id = ?", (cluster_id, face_id)
    )


def reassign_faces(conn: sqlite3.Connection, from_cluster: int, to_cluster: int) -> None:
    conn.execute("UPDATE faces SET cluster_id = ? WHERE cluster_id = ?", (to_cluster, from_cluster))


# --- clusters (people) ---


def create_cluster(
    conn: sqlite3.Connection,
    *,
    rep_face_id: int | None,
    size: int,
    centroid: bytes,
    now: float,
    label: str | None = None,
    pinned: int = 0,
) -> int:
    cur = conn.execute(
        "INSERT INTO clusters (label, pinned, rep_face_id, size, centroid, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (label, pinned, rep_face_id, size, centroid, now),
    )
    assert cur.lastrowid is not None
    return cur.lastrowid


def update_cluster(
    conn: sqlite3.Connection,
    cluster_id: int,
    *,
    now: float,
    size: int | None = None,
    rep_face_id: int | None = None,
    centroid: bytes | None = None,
) -> None:
    sets = ["updated_at = ?"]
    params: list[Any] = [now]
    for column, value in (("size", size), ("rep_face_id", rep_face_id), ("centroid", centroid)):
        if value is not None:
            sets.append(f"{column} = ?")
            params.append(value)
    params.append(cluster_id)
    conn.execute(f"UPDATE clusters SET {', '.join(sets)} WHERE id = ?", params)


def set_label(conn: sqlite3.Connection, cluster_id: int, label: str | None, now: float) -> None:
    """Rename a person; a non-empty label pins the cluster (survives re-clustering)."""
    conn.execute(
        "UPDATE clusters SET label = ?, pinned = ?, updated_at = ? WHERE id = ?",
        (label, 1 if label else 0, now, cluster_id),
    )


def delete_cluster(conn: sqlite3.Connection, cluster_id: int) -> None:
    conn.execute("DELETE FROM clusters WHERE id = ?", (cluster_id,))


def clusters_with_centroids(
    conn: sqlite3.Connection,
) -> list[tuple[int, bytes, int, int, int | None]]:
    """(id, centroid, size, pinned, rep_face_id) for every non-empty cluster."""
    rows = conn.execute(
        "SELECT id, centroid, size, pinned, rep_face_id FROM clusters WHERE centroid IS NOT NULL"
    ).fetchall()
    return [(int(r[0]), bytes(r[1]), int(r[2]), int(r[3]), r[4]) for r in rows]


def cluster_rep_rows(conn: sqlite3.Connection) -> list[tuple[int, int, int]]:
    """(cluster_id, rep_face_id, rep_embed_row) — anchors for incremental re-clustering."""
    rows = conn.execute(
        "SELECT c.id, c.rep_face_id, f.embed_row FROM clusters c "
        "JOIN faces f ON f.id = c.rep_face_id "
        "WHERE c.rep_face_id IS NOT NULL AND f.embed_row IS NOT NULL"
    ).fetchall()
    return [(int(r[0]), int(r[1]), int(r[2])) for r in rows]


def list_people(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """People with cover info, largest first."""
    rows = conn.execute(
        "SELECT c.id, c.label, c.pinned, c.size, c.rep_face_id, f.photo_id AS rep_photo_id "
        "FROM clusters c LEFT JOIN faces f ON f.id = c.rep_face_id "
        "WHERE c.size > 0 ORDER BY c.size DESC, c.id"
    ).fetchall()
    return [dict(r) for r in rows]


def get_person(conn: sqlite3.Connection, cluster_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT c.id, c.label, c.pinned, c.size, c.rep_face_id, f.photo_id AS rep_photo_id "
        "FROM clusters c LEFT JOIN faces f ON f.id = c.rep_face_id WHERE c.id = ?",
        (cluster_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def photos_for_person(conn: sqlite3.Connection, cluster_id: int, limit: int) -> list[int]:
    rows = conn.execute(
        "SELECT DISTINCT photo_id FROM faces WHERE cluster_id = ? ORDER BY photo_id DESC LIMIT ?",
        (cluster_id, limit),
    ).fetchall()
    return [int(r[0]) for r in rows]
