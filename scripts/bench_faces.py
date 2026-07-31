"""Phase 3 face benchmark — REAL SCRFD + ArcFace, logged to `benchmarks`.

Measures on this machine:
  * face_throughput (faces/s) — real SCRFD detect + ArcFace embed (CoreML)
  * cluster_purity — do the SAME real face, under brightness/contrast/blur changes,
    cluster together? Photometric augmentations keep geometry fixed, so a face's
    position labels its identity; purity = fraction of faces in a cluster whose
    majority identity matches. Uses insightface's bundled test image (no user data).

Run:  cd backend && uv run python ../scripts/bench_faces.py
"""

from __future__ import annotations

import platform
import tempfile
import time
from pathlib import Path

import cv2
import insightface
import numpy as np

from iris import benchmarks
from iris.db import apply_migrations, connect
from iris.faces.cluster import cluster_embeddings
from iris.faces.detector import FaceDetector

MODEL = "buffalo_l"


def _augment(img: np.ndarray) -> list[np.ndarray]:
    f = img.astype(np.float32)
    return [
        img,
        np.clip(f * 1.35, 0, 255).astype(np.uint8),  # brighter
        np.clip(f * 0.65, 0, 255).astype(np.uint8),  # darker
        np.clip((f - 128) * 1.4 + 128, 0, 255).astype(np.uint8),  # higher contrast
        cv2.GaussianBlur(img, (5, 5), 0),  # soft blur
        np.clip(((f / 255.0) ** 0.6) * 255.0, 0, 255).astype(np.uint8),  # gamma
    ]


def _center(bbox: tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x + w / 2, y + h / 2], dtype=np.float64)


def main() -> None:
    detector = FaceDetector.load(MODEL, det_size=640, min_size=24, min_det_score=0.4)
    base_img = insightface.data.get_image("t1")  # BGR, several faces

    base = detector.detect(base_img)
    base_centers = [_center(f.bbox) for f in base]
    print(f"base image: {len(base)} identities")

    embeddings: list[np.ndarray] = []
    identities: list[int] = []
    total_faces = 0
    detect_secs = 0.0
    for variant in _augment(base_img):
        t0 = time.perf_counter()
        faces = detector.detect(variant)
        detect_secs += time.perf_counter() - t0
        total_faces += len(faces)
        for face in faces:
            dists = [np.linalg.norm(_center(face.bbox) - c) for c in base_centers]
            nearest = int(np.argmin(dists))
            if dists[nearest] < 0.05:  # normalized distance -> same position -> same person
                embeddings.append(face.embedding)
                identities.append(nearest)

    face_throughput = total_faces / detect_secs

    matrix = np.stack(embeddings).astype(np.float32)
    ids = list(range(len(embeddings)))
    clusters = cluster_embeddings(ids, matrix, k=10, edge_threshold=0.5)
    by_cluster: dict[int, list[int]] = {}
    for face_idx, cluster in clusters.items():
        by_cluster.setdefault(cluster, []).append(identities[face_idx])
    correct = sum(max(np.bincount(members).tolist()) for members in by_cluster.values())
    purity = correct / len(embeddings)

    context = {
        "dataset": "insightface-t1 + photometric aug", "identities": len(base),
        "faces": len(embeddings), "clusters": len(by_cluster),
        "model": MODEL, "machine": platform.platform(),
    }
    work = Path(tempfile.mkdtemp(prefix="iris-facebench-"))
    conn = connect(work / "bench.sqlite")
    apply_migrations(conn)
    benchmarks.record_benchmark(conn, name="face_throughput", value=round(face_throughput, 2),
                                unit="faces/s", n=total_faces, phase="3", context=context)
    benchmarks.record_benchmark(conn, name="cluster_purity", value=round(purity, 4),
                                unit="fraction", n=len(embeddings), phase="3", context=context)
    conn.commit()
    conn.close()

    print("\n=== Phase 3 face benchmark ===")
    print(f"  identities:      {len(base)}  | faces embedded: {len(embeddings)}")
    print(f"  face_throughput: {face_throughput:.1f} faces/s (SCRFD+ArcFace, CoreML)")
    print(f"  clusters formed: {len(by_cluster)}")
    print(f"  cluster_purity:  {purity:.3f}")
    print(f"\n  logged 2 rows to benchmarks at {work / 'bench.sqlite'}")


if __name__ == "__main__":
    main()
