"""Phase 2 embed + search benchmark — REAL CLIP model, logged to `benchmarks`.

Measures, from an actual run on this machine:
  * embed_throughput (img/s) — real CLIP vision encoder on the CoreML EP, batched
  * search_p50 / search_p95 (ms) — text query -> vector -> HNSW query, end to end
  * recall_at_10 — HNSW top-10 vs. exact brute-force top-10 agreement

Dataset is synthetic noise images (labelled as such); the real ~100k library run is a
Phase 6 task. Numbers are never fabricated — they come from this run.

Run:  cd backend && uv run python ../scripts/bench_embed_search.py --count 800
"""

from __future__ import annotations

import argparse
import platform
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from iris import benchmarks
from iris.db import apply_migrations, connect
from iris.embeddings.clip import CLIPEmbedder
from iris.embeddings.index import VectorIndex
from iris.embeddings.store import VectorStore

MODEL_ID = "Xenova/clip-vit-base-patch32"
DIM = 512
QUERIES = [
    "a dog on a beach", "a red car", "mountains at sunset", "a plate of food",
    "a person smiling", "city street at night", "a cat on a sofa", "green forest",
    "snow covered trees", "a cup of coffee",
]


def main() -> None:
    parser = argparse.ArgumentParser(prog="bench_embed_search")
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--queries", type=int, default=50)
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="iris-embbench-"))
    rng = np.random.default_rng(7)
    print(f"generating {args.count} synthetic images…", flush=True)
    images = [
        Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), "RGB")
        for _ in range(args.count)
    ]

    print("loading CLIP model (cached after first run)…", flush=True)
    embedder = CLIPEmbedder.load(work / "models", MODEL_ID, DIM)

    # --- embed throughput (real CoreML vision encoder) ---
    vectors = np.empty((args.count, DIM), dtype=np.float32)
    start = time.perf_counter()
    for i in range(0, args.count, args.batch):
        chunk = images[i : i + args.batch]
        vectors[i : i + len(chunk)] = embedder.embed_images(chunk)
    embed_secs = time.perf_counter() - start
    embed_throughput = args.count / embed_secs

    # --- build store + index ---
    store = VectorStore(work / "clip.f32", DIM, MODEL_ID)
    store.append(vectors)
    ids = np.arange(args.count, dtype=np.int64)
    index = VectorIndex(DIM)
    index.build(vectors, ids)

    # --- search latency (text -> vector -> HNSW), end to end ---
    latencies_ms: list[float] = []
    for i in range(args.queries):
        prompt = QUERIES[i % len(QUERIES)]
        t0 = time.perf_counter()
        qv = embedder.embed_texts([prompt])[0]
        index.query(qv, k=10)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    search_p50 = statistics.median(latencies_ms)
    search_p95 = float(np.percentile(latencies_ms, 95))

    # --- recall@10 vs exact brute force (use image vectors as queries) ---
    matrix = store.matrix()
    q_idx = rng.choice(args.count, size=min(args.queries, args.count), replace=False)
    hits = 0
    for qi in q_idx:
        qv = vectors[qi]
        brute = set(np.argsort(-(matrix @ qv))[:10].tolist())
        labels, _ = index.query(qv, k=10, ef=200)
        hits += len(brute & set(labels))
    recall_at_10 = hits / (len(q_idx) * 10)

    context = {
        "dataset": "synthetic-noise", "count": args.count, "batch": args.batch,
        "model": MODEL_ID, "vision_ep": "CoreML", "text_ep": "CPU",
        "machine": platform.platform(),
    }
    conn = connect(work / "bench.sqlite")
    apply_migrations(conn)
    for name, value, unit in [
        ("embed_throughput", round(embed_throughput, 2), "img/s"),
        ("search_p50", round(search_p50, 2), "ms"),
        ("search_p95", round(search_p95, 2), "ms"),
        ("recall_at_10", round(recall_at_10, 4), "fraction"),
    ]:
        benchmarks.record_benchmark(conn, name=name, value=value, unit=unit, n=args.count,
                                    phase="2", context=context)
    conn.commit()
    conn.close()

    print("\n=== Phase 2 embed + search benchmark (synthetic dataset) ===")
    print(f"  images:            {args.count} (256x256), batch {args.batch}")
    print(f"  embed_throughput:  {embed_throughput:.1f} img/s  ({embed_secs:.2f}s, vision=CoreML)")
    print(f"  search_p50/p95:    {search_p50:.2f} / {search_p95:.2f} ms  (text=CPU + HNSW)")
    print(f"  recall@10 vs brute:{recall_at_10:.3f}")
    print(f"\n  logged 4 rows to benchmarks at {work / 'bench.sqlite'}")


if __name__ == "__main__":
    main()
