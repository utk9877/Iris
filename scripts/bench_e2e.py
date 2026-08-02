"""Phase 6 end-to-end benchmark harness — REAL full-library run, logged to `benchmarks`.

Measures on YOUR machine against YOUR library (no fabricated numbers, per CLAUDE.md):
  * e2e_ingest       — wall-clock throughput (photos/second) for a complete ingest of a
                       root folder: scan -> hash/exif -> thumb/phash -> score -> (embed) ->
                       (faces) -> (ocr) -> grouping.
  * peak_rss         — peak resident set size (bytes) reached during the ingest, via
                       getrusage(RUSAGE_SELF).ru_maxrss (a high-watermark).
  * search_p95_full  — 95th-percentile latency (ms) of semantic search over the full index
                       (only with --embed, which loads CLIP; needs the `embed` extra).

The headline **~100k / 500 GB** numbers (PHASES Phase 6 done-when) come from running this
on the real library:

    cd backend && uv run --extra embed --extra faces --extra ocr --extra geo \
        python ../scripts/bench_e2e.py --root /path/to/Photos --embed --data-dir /tmp/iris-e2e

Without --root it generates a tiny synthetic set so the harness itself can be smoke-tested
(the resulting numbers are trivially small — not the headline figures).
"""

from __future__ import annotations

import argparse
import os
import platform
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path


def _rss_bytes() -> int:
    """Peak RSS so far, normalized to bytes (macOS reports bytes, Linux kibibytes)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru if sys.platform == "darwin" else ru * 1024)


def _generate(root: Path, n: int) -> None:
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (256, 192), (i * 7 % 256, i * 13 % 256, i * 29 % 256)).save(
            root / f"img{i:04d}.png"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="bench_e2e")
    parser.add_argument("--root", help="library folder to ingest (default: generate a smoke set)")
    parser.add_argument("--data-dir", help="app data dir (default: a temp dir)")
    parser.add_argument("--embed", action="store_true", help="run CLIP embed + search timing")
    parser.add_argument("--faces", action="store_true", help="run the faces stage")
    parser.add_argument("--ocr", action="store_true", help="run the OCR stage")
    parser.add_argument("--smoke-n", type=int, default=24, help="synthetic photo count")
    parser.add_argument(
        "--queries",
        default="beach,mountains,birthday cake,city street at night,dog",
        help="comma-separated search queries for search_p95_full",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(tempfile.mkdtemp(prefix="iris-e2e-"))
    os.environ["IRIS_DATA_DIR"] = str(data_dir)

    # Import after IRIS_DATA_DIR is set so settings resolve to the isolated dir.
    from iris.config import get_settings
    from iris.db import apply_migrations, connect
    from iris.db import library as library_db
    from iris.db import photos as photos_db
    from iris.embeddings.service import EmbeddingService
    from iris.faces.service import FacesService
    from iris.grouping.service import GroupingService
    from iris.ingest.orchestrator import IngestManager
    from iris.ocr.service import OcrService

    get_settings.cache_clear()
    settings = get_settings()
    settings.ensure_dirs()

    root = Path(args.root) if args.root else data_dir / "smoke-src"
    if not args.root:
        print(f"no --root given; generating {args.smoke_n} synthetic photos (smoke run)")
        _generate(root, args.smoke_n)

    conn = connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    apply_migrations(conn)
    library_db.add_root(conn, str(root), time.time())
    conn.close()

    embeddings = EmbeddingService(settings) if args.embed else None
    grouping = GroupingService(settings, embeddings)
    manager = IngestManager(
        settings,
        embeddings,
        FacesService(settings) if args.faces else None,
        OcrService(settings) if args.ocr else None,
        grouping,
    )

    t0 = time.perf_counter()
    manager.start()
    while manager.is_running():
        time.sleep(0.2)
    elapsed = time.perf_counter() - t0
    peak_rss = _rss_bytes()

    conn = connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    n_photos = photos_db.count_photos(conn)
    throughput = n_photos / elapsed if elapsed > 0 else 0.0

    # search_p95_full — only meaningful with a real index.
    p95_ms: float | None = None
    latencies: list[float] = []
    if embeddings is not None and n_photos > 0:
        from iris import search as search_engine

        embedder = embeddings.embedder()
        embeddings.ensure_index()
        for q in [s.strip() for s in args.queries.split(",") if s.strip()]:
            vec = embedder.embed_texts([q])[0]
            s = time.perf_counter()
            search_engine.semantic_search(
                embeddings, conn, vec, k=50,
                brute_max=settings.search_brute_max, filtered_max=settings.search_filtered_max,
            )
            latencies.append((time.perf_counter() - s) * 1000.0)
        if latencies:
            p95_ms = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 20 else max(
                latencies
            )

    from iris import benchmarks

    machine = platform.platform()
    context = {"photos": n_photos, "root": str(root), "embed": bool(embeddings),
               "faces": args.faces, "ocr": args.ocr, "machine": machine}
    benchmarks.record_benchmark(conn, name="e2e_ingest", value=round(throughput, 2),
                                unit="photos/s", n=n_photos, phase="6", context=context)
    benchmarks.record_benchmark(conn, name="peak_rss", value=float(peak_rss),
                                unit="bytes", n=n_photos, phase="6", context=context)
    if p95_ms is not None:
        benchmarks.record_benchmark(conn, name="search_p95_full", value=round(p95_ms, 3),
                                    unit="ms", n=len(latencies), phase="6", context=context)
    conn.commit()
    conn.close()

    print("\n=== Phase 6 end-to-end benchmark ===")
    print(f"  photos ingested:  {n_photos}")
    print(f"  e2e_ingest:       {throughput:.2f} photos/s  ({elapsed:.1f}s wall)")
    print(f"  peak_rss:         {peak_rss / 1024**2:.0f} MiB")
    print(f"  search_p95_full:  {f'{p95_ms:.2f} ms' if p95_ms is not None else 'skipped (no --embed)'}")
    print(f"\n  logged to benchmarks at {settings.db_path}")
    if not args.root:
        print("  NOTE: smoke run on synthetic data — run with --root <library> for headline numbers.")


if __name__ == "__main__":
    main()
