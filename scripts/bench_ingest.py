"""Phase 1 ingest benchmark — runs a REAL ingest and logs results to `benchmarks`.

Per the CLAUDE.md standing instruction, numbers must come from an actual run. This
generates a synthetic image set (clearly labelled as such in the benchmark context)
and measures end-to-end ingest throughput plus per-image decode latency. The real
~100k / 500 GB library run is a Phase 6 task; these synthetic numbers only prove the
pipeline works and give a baseline on this machine.

Run:  cd backend && uv run python ../scripts/bench_ingest.py --count 300
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image


def _generate_images(folder: Path, count: int, size: tuple[int, int]) -> None:
    """Write `count` random-noise JPEGs (non-trivial to decode, unlike solids)."""
    rng = np.random.default_rng(1234)
    width, height = size
    for i in range(count):
        noise = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(noise, mode="RGB").save(folder / f"img_{i:06d}.jpg", quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(prog="bench_ingest")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--decode-sample", type=int, default=40)
    args = parser.parse_args()
    size = (args.width, args.height)

    workdir = Path(tempfile.mkdtemp(prefix="iris-bench-"))
    data_dir = workdir / "data"
    source_dir = workdir / "source"
    source_dir.mkdir(parents=True)
    os.environ["IRIS_DATA_DIR"] = str(data_dir)

    # Import after IRIS_DATA_DIR is set so settings resolve to the temp dir.
    from iris import benchmarks
    from iris.config import get_settings
    from iris.db import apply_migrations, connect
    from iris.db import library as library_db
    from iris.ingest.orchestrator import IngestManager
    from iris.ingest.thumbnails import render_thumbnail

    get_settings.cache_clear()
    settings = get_settings()
    settings.ensure_dirs()

    print(f"generating {args.count} synthetic {size[0]}x{size[1]} JPEGs ...", flush=True)
    _generate_images(source_dir, args.count, size)
    total_bytes = sum(p.stat().st_size for p in source_dir.iterdir())
    avg_kb = total_bytes / args.count / 1024

    conn = connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    apply_migrations(conn)
    library_db.add_root(conn, str(source_dir), time.time())
    conn.close()

    # --- End-to-end ingest throughput ---
    workers = settings.resolved_decode_workers()
    print(f"ingesting with {workers} decode workers ...", flush=True)
    manager = IngestManager(settings)
    start = time.perf_counter()
    manager.start()
    while manager.is_running():
        time.sleep(0.05)
    elapsed = time.perf_counter() - start
    throughput = args.count / elapsed
    job = manager.status()["job"]
    assert job is not None and job["state"] == "done", job

    # --- Per-image decode latency (single-process, isolates decode+thumb+phash) ---
    sample = sorted(source_dir.iterdir())[: args.decode_sample]
    timings_ms: list[float] = []
    for path in sample:
        t0 = time.perf_counter()
        render_thumbnail(0, str(path), "bench" + path.stem, str(data_dir / "benchthumbs"))
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    decode_p50 = statistics.median(timings_ms)

    context = {
        "dataset": "synthetic-noise-jpeg",
        "count": args.count,
        "image_px": f"{size[0]}x{size[1]}",
        "avg_kb": round(avg_kb, 1),
        "decode_workers": workers,
        "machine": platform.platform(),
    }

    conn = connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    benchmarks.record_benchmark(
        conn, name="ingest_throughput", value=round(throughput, 2), unit="img/s",
        n=args.count, phase="1", context=context,
    )
    benchmarks.record_benchmark(
        conn, name="decode_p50", value=round(decode_p50, 2), unit="ms",
        n=len(timings_ms), phase="1", context=context,
    )
    conn.commit()
    conn.close()

    print("\n=== Phase 1 ingest benchmark (synthetic dataset) ===")
    print(f"  images:            {args.count}  (~{avg_kb:.0f} KB each, {size[0]}x{size[1]})")
    print(f"  decode workers:    {workers}")
    print(f"  ingest_throughput: {throughput:.1f} img/s  ({elapsed:.2f}s total)")
    print(f"  decode_p50:        {decode_p50:.1f} ms/img")
    print(f"  errored:           {job['errored']}")
    print(f"\n  logged to benchmarks table at {settings.db_path}")


if __name__ == "__main__":
    main()
