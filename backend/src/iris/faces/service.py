"""Face clustering service: incremental assignment + pending-pool re-clustering (§6).

Owns the ``faces.f32`` memmap. New faces are first assigned to the nearest existing
person by centroid cosine; the leftovers accumulate in a pending pool that is
re-clustered (Chinese Whispers over pending faces + existing cluster reps) when it
grows past a threshold, so labeled people are never rebuilt from scratch.
"""

from __future__ import annotations

import logging
import sqlite3
import time

import numpy as np

from iris.config import Settings
from iris.db import faces as faces_db
from iris.embeddings.store import VectorStore
from iris.faces.cluster import Vectors, cluster_embeddings
from iris.faces.detector import FaceDetector

logger = logging.getLogger("iris.faces")


class FacesService:
    def __init__(self, settings: Settings, *, detector: FaceDetector | None = None) -> None:
        self._settings = settings
        self._store = VectorStore(
            settings.embeddings_dir / "faces.f32", settings.face_dim, settings.face_model
        )
        self._detector = detector

    @property
    def store(self) -> VectorStore:
        return self._store

    def detector(self) -> FaceDetector:
        """Lazily load the insightface detector/embedder (downloads on first use)."""
        if self._detector is None:
            logger.info("loading face model %s (first use may download)", self._settings.face_model)
            self._detector = FaceDetector.load(
                self._settings.face_model,
                det_size=self._settings.face_det_size,
                min_size=self._settings.face_min_size,
                min_det_score=self._settings.face_min_det_score,
            )
        return self._detector

    def compact(self, conn: sqlite3.Connection) -> dict[str, int]:
        """Reclaim orphaned face vectors and renumber ``faces.embed_row`` (ARCHITECTURE §3).

        Faces reference the store by ``embed_row`` and cluster centroids are stored as raw
        bytes (not row refs), so compaction only rewrites the memmap and repoints the face
        rows — clusters are untouched. Run only when ingest is idle.
        """
        pairs = faces_db.all_face_embed_rows(conn)  # (embed_row, face_id) by embed_row
        before = self._store.count
        mapping = self._store.compact([old for old, _ in pairs])
        conn.execute("BEGIN")
        try:
            for old_row, face_id in pairs:
                faces_db.set_face_embed_row(conn, face_id, mapping[old_row])
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return {"before": before, "after": self._store.count, "reclaimed": before - len(pairs)}

    # ------------------------------------------------------------- centroids
    @staticmethod
    def _to_bytes(vec: Vectors) -> bytes:
        return np.ascontiguousarray(vec, dtype=np.float32).tobytes()

    def _normalized_mean(self, embed_rows: list[int]) -> Vectors:
        mean = self._store.get_rows(embed_rows).mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return np.asarray(mean / norm if norm > 0 else mean, dtype=np.float32)

    def _recompute_cluster(self, conn: sqlite3.Connection, cluster_id: int, now: float) -> None:
        rows = faces_db.cluster_face_rows(conn, cluster_id)
        if not rows:
            faces_db.delete_cluster(conn, cluster_id)
            return
        embed_rows = [r[1] for r in rows]
        rep_face_id = max(rows, key=lambda r: r[2])[0]  # highest quality
        centroid = self._normalized_mean(embed_rows)
        faces_db.update_cluster(
            conn,
            cluster_id,
            now=now,
            size=len(rows),
            rep_face_id=rep_face_id,
            centroid=self._to_bytes(centroid),
        )

    # ------------------------------------------------------------- assignment
    def assign_pending(self, conn: sqlite3.Connection) -> int:
        """Assign pending faces to the nearest existing person (centroid cosine)."""
        clusters = faces_db.clusters_with_centroids(conn)
        pending = faces_db.pending_faces(conn)
        if not clusters or not pending:
            return 0
        cluster_ids = [c[0] for c in clusters]
        centroids = np.stack([np.frombuffer(c[1], dtype=np.float32) for c in clusters])
        threshold = self._settings.face_assign_threshold

        assigned = 0
        touched: set[int] = set()
        conn.execute("BEGIN")
        try:
            for face_id, embed_row, _quality in pending:
                vector = self._store.get_rows([embed_row])[0]
                sims = centroids @ vector
                best = int(np.argmax(sims))
                if float(sims[best]) >= threshold:
                    faces_db.assign_face(conn, face_id, cluster_ids[best])
                    touched.add(cluster_ids[best])
                    assigned += 1
            for cluster_id in touched:
                self._recompute_cluster(conn, cluster_id, time.time())
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return assigned

    # ----------------------------------------------------------- re-cluster
    def recluster_pending(self, conn: sqlite3.Connection) -> int:
        """Re-cluster the pending pool together with existing cluster reps.

        Returns the number of clusters created or merged into. Pending faces that land
        in a group with an existing cluster's rep join that cluster (labels preserved);
        pure-pending groups become new clusters; groups bridging two clusters merge them.
        """
        pending = faces_db.pending_faces(conn)
        if not pending:
            return 0
        reps = faces_db.cluster_rep_rows(conn)  # (cluster_id, rep_face_id, embed_row)
        rep_of_face = {rep_face_id: cluster_id for cluster_id, rep_face_id, _ in reps}
        sizes = {c[0]: c[2] for c in faces_db.clusters_with_centroids(conn)}

        node_ids = [f[0] for f in pending] + [r[1] for r in reps]
        embed_rows = [f[1] for f in pending] + [r[2] for r in reps]
        quality = {f[0]: f[2] for f in pending}
        matrix = self._store.get_rows(embed_rows)

        groups = cluster_embeddings(
            node_ids,
            matrix,
            k=self._settings.face_cluster_k,
            edge_threshold=self._settings.face_edge_threshold,
        )
        members: dict[int, list[int]] = {}
        for node, label in groups.items():
            members.setdefault(label, []).append(node)

        now = time.time()
        touched: set[int] = set()
        conn.execute("BEGIN")
        try:
            for group in members.values():
                rep_clusters = {rep_of_face[n] for n in group if n in rep_of_face}
                pending_faces = [n for n in group if n not in rep_of_face]
                if rep_clusters:
                    target = max(rep_clusters, key=lambda cid: sizes.get(cid, 0))
                    for other in rep_clusters - {target}:
                        faces_db.reassign_faces(conn, other, target)
                        faces_db.delete_cluster(conn, other)
                    for face_id in pending_faces:
                        faces_db.assign_face(conn, face_id, target)
                    touched.add(target)
                elif pending_faces:
                    rep = max(pending_faces, key=lambda fid: quality.get(fid, 0.0))
                    cluster_id = faces_db.create_cluster(
                        conn, rep_face_id=rep, size=0, centroid=b"", now=now
                    )
                    for face_id in pending_faces:
                        faces_db.assign_face(conn, face_id, cluster_id)
                    touched.add(cluster_id)
            for cluster_id in touched:
                self._recompute_cluster(conn, cluster_id, now)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return len(touched)

    def _reset_unpinned(self, conn: sqlite3.Connection) -> None:
        """Send every face in an unlabeled cluster back to the pending pool.

        Labeled (pinned) people are kept intact — their faces stay assigned and their
        reps anchor re-clustering — so a full re-cluster never destroys manual naming.
        """
        conn.execute("BEGIN")
        try:
            conn.execute(
                "UPDATE faces SET assigned = 0, cluster_id = NULL WHERE cluster_id IN "
                "(SELECT id FROM clusters WHERE pinned = 0)"
            )
            conn.execute("DELETE FROM clusters WHERE pinned = 0")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def update_people(self, conn: sqlite3.Connection, *, force: bool = False) -> None:
        """Post-ingest hook: assign to existing people, then re-cluster the pool if large.

        ``force`` (the "Re-cluster" action) first dissolves all unlabeled people so the
        whole library is re-clustered with the current thresholds.
        """
        if force:
            self._reset_unpinned(conn)
        self.assign_pending(conn)
        pool = faces_db.count_pending_pool(conn)
        has_clusters = bool(faces_db.clusters_with_centroids(conn))
        if force or pool >= self._settings.face_pool_threshold or (pool > 0 and not has_clusters):
            self.recluster_pending(conn)

    # ------------------------------------------------------------- edits (UI)
    def _create_from_faces(
        self, conn: sqlite3.Connection, face_rows: list[tuple[int, int, float]], now: float
    ) -> int:
        rep = max(face_rows, key=lambda r: r[2])[0]
        cluster_id = faces_db.create_cluster(conn, rep_face_id=rep, size=0, centroid=b"", now=now)
        for face_id, _row, _q in face_rows:
            faces_db.assign_face(conn, face_id, cluster_id)
        self._recompute_cluster(conn, cluster_id, now)
        return cluster_id

    def merge_clusters(
        self, conn: sqlite3.Connection, target_id: int, other_ids: list[int]
    ) -> None:
        """Merge ``other_ids`` into ``target_id`` (target keeps its label)."""
        now = time.time()
        conn.execute("BEGIN")
        try:
            for other in other_ids:
                if other == target_id:
                    continue
                faces_db.reassign_faces(conn, other, target_id)
                faces_db.delete_cluster(conn, other)
            self._recompute_cluster(conn, target_id, now)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def split_cluster(self, conn: sqlite3.Connection, cluster_id: int) -> int:
        """Re-cluster one person's faces in isolation with a stricter threshold.

        Useful when two people were over-merged; returns the number of resulting people.
        """
        rows = faces_db.cluster_face_rows(conn, cluster_id)
        if len(rows) <= 1:
            return 0
        now = time.time()
        ids = [r[0] for r in rows]
        matrix = self._store.get_rows([r[1] for r in rows])
        strict = min(0.75, self._settings.face_edge_threshold + 0.15)
        groups = cluster_embeddings(
            ids, matrix, k=self._settings.face_cluster_k, edge_threshold=strict
        )
        by_group: dict[int, list[tuple[int, int, float]]] = {}
        row_by_id = {r[0]: r for r in rows}
        for face_id, label in groups.items():
            by_group.setdefault(label, []).append(row_by_id[face_id])

        conn.execute("BEGIN")
        try:
            faces_db.delete_cluster(conn, cluster_id)
            for group_rows in by_group.values():
                self._create_from_faces(conn, group_rows, now)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return len(by_group)
