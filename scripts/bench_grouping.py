"""Phase 4 grouping benchmark — REAL groupers on generated ground-truth, logged to `benchmarks`.

Measures on this machine (no user data, no fabricated numbers):
  * grouping_f1 — pairwise F1 of the REAL near-duplicate grouper. We render N distinct
    base scenes, make k near-identical variants of each (JPEG re-encode + tiny brightness
    shift), compute the REAL perceptual hash of every file, and run the actual
    `find_near_dups`. Ground truth = "same base scene". F1 is over co-membership pairs.
  * grouping_throughput — photos/second through phash + the grouping pass.
  * event_recall — the REAL event segmenter's boundary recall on a synthetic timeline
    with known day/trip gaps.

Run:  cd backend && uv run python ../scripts/bench_grouping.py
"""

from __future__ import annotations

import itertools
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from iris import benchmarks
from iris.config import get_settings
from iris.db import apply_migrations, connect
from iris.grouping.events import segment_events
from iris.grouping.near_dup import find_near_dups
from iris.ingest.phash import perceptual_hash

N_SCENES = 12
VARIANTS = 4
RNG = np.random.default_rng(7)


def _base_scene() -> Image.Image:
    """A structured random scene (blocky, so JPEG survives it) — distinct per call."""
    small = RNG.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((256, 256), Image.Resampling.NEAREST)


def _variants(img: Image.Image, out_dir: Path, scene: int) -> list[Path]:
    paths: list[Path] = []
    for v in range(VARIANTS):
        arr = np.asarray(img, dtype=np.int16) + (v - VARIANTS // 2) * 4  # ±brightness
        variant = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        path = out_dir / f"scene{scene}_v{v}.jpg"
        variant.save(path, quality=70 + v * 7)  # differing JPEG quality
        paths.append(path)
    return paths


def _row(pid: int, phash: int, sort_at: float) -> dict[str, object]:
    return {
        "id": pid,
        "phash": phash,
        "sort_at": sort_at,
        "camera_model": "Cam",
        "embed_row": None,
        "gps_lat": None,
        "gps_lon": None,
        "width": 256,
        "height": 256,
    }


def _pairwise_f1(labels: dict[int, int], truth: dict[int, int]) -> tuple[float, float, float]:
    ids = list(labels)
    tp = fp = fn = 0
    for a, b in itertools.combinations(ids, 2):
        same_pred = labels[a] == labels[b]
        same_true = truth[a] == truth[b]
        if same_pred and same_true:
            tp += 1
        elif same_pred and not same_true:
            fp += 1
        elif not same_pred and same_true:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    settings = get_settings()
    work = Path(tempfile.mkdtemp(prefix="iris-groupbench-"))
    rows: list[dict[str, object]] = []
    truth: dict[int, int] = {}

    t0 = time.perf_counter()
    pid = 0
    for scene in range(N_SCENES):
        for path in _variants(_base_scene(), work, scene):
            phash = perceptual_hash(Image.open(path))
            rows.append(_row(pid, phash, sort_at=float(pid)))
            truth[pid] = scene
            pid += 1
    phash_secs = time.perf_counter() - t0

    t1 = time.perf_counter()
    components = find_near_dups(
        rows, hamming_max=settings.near_dup_hamming, cosine_min=0.0
    )
    group_secs = time.perf_counter() - t1

    labels = {r["id"]: -r["id"] for r in rows}  # singletons -> unique negative labels
    for gi, comp in enumerate(components):
        for r in comp:
            labels[r["id"]] = gi
    precision, recall, f1 = _pairwise_f1(labels, truth)
    throughput = len(rows) / (phash_secs + group_secs)

    # Event segmenter recall on a synthetic 3-day timeline (gaps > 4h between days).
    day = 86400.0
    ev_rows = []
    ev_truth_boundaries = set()
    ts = 0.0
    for d in range(3):
        for _ in range(5):
            ev_rows.append(_row(len(ev_rows), 0, ts))
            ts += 600.0  # 10 min apart within a day
        if d < 2:
            ev_truth_boundaries.add(len(ev_rows))  # boundary index
            ts += 6 * 3600.0  # 6h gap -> new day
    events = segment_events(
        ev_rows, gap_seconds=settings.event_gap_seconds, gps_km=settings.event_gps_km
    )
    got_boundaries = set(itertools.accumulate(len(e) for e in events))
    event_recall = len(ev_truth_boundaries & got_boundaries) / len(ev_truth_boundaries)

    context = {
        "scenes": N_SCENES, "variants": VARIANTS, "photos": len(rows),
        "hamming_max": settings.near_dup_hamming, "precision": round(precision, 4),
        "recall": round(recall, 4), "machine": platform.platform(),
    }
    conn = connect(work / "bench.sqlite")
    apply_migrations(conn)
    benchmarks.record_benchmark(conn, name="grouping_f1", value=round(f1, 4),
                                unit="f1", n=len(rows), phase="4", context=context)
    benchmarks.record_benchmark(conn, name="grouping_throughput", value=round(throughput, 1),
                                unit="photos/s", n=len(rows), phase="4", context=context)
    benchmarks.record_benchmark(conn, name="event_recall", value=round(event_recall, 4),
                                unit="fraction", n=len(ev_rows), phase="4", context=context)
    conn.commit()
    conn.close()

    print("\n=== Phase 4 grouping benchmark ===")
    print(f"  photos:               {len(rows)} ({N_SCENES} scenes x {VARIANTS} variants)")
    print(f"  near_dup precision:   {precision:.3f}")
    print(f"  near_dup recall:      {recall:.3f}")
    print(f"  grouping_f1:          {f1:.3f}")
    print(f"  grouping_throughput:  {throughput:.1f} photos/s")
    print(f"  event_recall:         {event_recall:.3f}")
    print(f"\n  logged 3 rows to benchmarks at {work / 'bench.sqlite'}")


if __name__ == "__main__":
    main()
