"""Phase 5 triage benchmark — REAL scorers + REAL triage pipeline, logged to `benchmarks`.

Measures on this machine (no user data, no fabricated numbers):
  * triage_overlap — we render N distinct "hero" scenes (sharp + colorful) and, for each,
    a few degraded near-duplicate variants (blurred + desaturated). The REAL aesthetic/
    quality scorers score every file; the REAL near-dup grouper (perceptual hash) clusters
    each hero with its variants. Ground-truth "best of cluster" = the hero (it is objectively
    the sharpest/most colorful frame we drew). We run the REAL `story-ready` triage over the
    whole event and measure the fraction of heroes that land in the top-N shortlist.
  * triage_throughput — photos/second through the full four-score + normalize + MMR pipeline.

CLIP vectors are stand-ins (deterministic per-scene unit directions in a real
EmbeddingService store) — triage only reads stored vectors, so no CLIP model is needed.

Run:  cd backend && uv run python ../scripts/bench_triage.py
"""

from __future__ import annotations

import platform
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from iris import benchmarks
from iris.config import get_settings
from iris.db import apply_migrations, connect
from iris.db.groups import GroupSpec, replace_groups
from iris.embeddings.service import EmbeddingService
from iris.grouping.near_dup import find_near_dups
from iris.ingest.phash import perceptual_hash
from iris.scoring import score_image
from iris.triage import get_preset
from iris.triage.service import TriageService

N_SCENES = 15
VARIANTS = 3  # degraded near-duplicates per hero
RNG = np.random.default_rng(11)


def _hero(scene: int) -> Image.Image:
    """A sharp, colorful, structured scene — distinct per call."""
    small = RNG.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((256, 256), Image.Resampling.NEAREST)


def _degrade(img: Image.Image) -> Image.Image:
    """Blur (downscale round-trip) + desaturate → a lower-quality near-duplicate."""
    small = img.resize((32, 32), Image.Resampling.BILINEAR).resize(
        (256, 256), Image.Resampling.BILINEAR
    )
    arr = np.asarray(small, dtype=np.float64)
    grey = arr.mean(axis=2, keepdims=True)
    muted = (0.4 * arr + 0.6 * grey).clip(0, 255).astype(np.uint8)  # pull toward grey
    return Image.fromarray(muted)


def _seed_photo(conn, pid: int, sort_at: float, aesthetic: float, quality: float,
                embed_row: int, phash: int) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO photos (id, path, dir, filename, ext, size_bytes, mtime, sort_at, "
        "aesthetic, quality, embed_row, phash, width, height, thumb_at, scored_at, missing, "
        "created_at, updated_at) VALUES (?, ?, '/b', ?, '.png', 1, ?, ?, ?, ?, ?, ?, 256, 256, "
        "?, ?, 0, ?, ?)",
        (pid, f"/b/p{pid}.png", f"p{pid}.png", now, sort_at, aesthetic, quality, embed_row,
         phash, now, now, now, now),
    )


def main() -> None:
    settings = get_settings()
    work = Path(tempfile.mkdtemp(prefix="iris-triagebench-"))
    settings.data_dir = work  # isolate the store/db under the temp dir

    service = EmbeddingService(settings)
    conn = connect(work / "bench.sqlite", busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    apply_migrations(conn)

    rows: list[dict[str, object]] = []
    hero_ids: list[int] = []
    dim = settings.embed_dim
    pid = 0
    for scene in range(N_SCENES):
        direction = np.zeros(dim, dtype=np.float32)
        direction[scene % dim] = 1.0  # each scene occupies its own CLIP direction
        hero = _hero(scene)
        frames = [(hero, True)] + [(_degrade(hero), False) for _ in range(VARIANTS)]
        for img, is_hero in frames:
            vec = direction.reshape(1, -1)
            embed_row = service.store.append(vec)
            phash = perceptual_hash(img)
            scores = score_image(img)
            _seed_photo(conn, pid, float(pid), scores.aesthetic, scores.quality, embed_row, phash)
            rows.append({"id": pid, "phash": phash, "embed_row": embed_row,
                         "width": 256, "height": 256})
            if is_hero:
                hero_ids.append(pid)
            pid += 1

    # Real near-dup grouping (phash union-find) + an event over everything.
    components = find_near_dups(rows, hamming_max=settings.near_dup_hamming, cosine_min=0.0)
    replace_groups(
        conn,
        "near_dup",
        [
            GroupSpec(kind="near_dup", members=[(int(r["id"]), float(i)) for i, r in enumerate(c)])
            for c in components
        ],
    )
    replace_groups(
        conn,
        "event",
        [GroupSpec(kind="event", members=[(int(r["id"]), float(i)) for i, r in enumerate(rows)])],
    )
    event_id = int(conn.execute("SELECT id FROM groups WHERE kind = 'event'").fetchone()[0])

    triage = TriageService(settings, service)
    # Timed runs (repeat for a stable throughput measurement).
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        result = triage.triage(
            conn, preset=get_preset("story-ready"), group_id=event_id, limit=N_SCENES
        )
    elapsed = time.perf_counter() - t0
    throughput = (reps * len(rows)) / elapsed

    picked = {it.photo_id for it in result.items}
    overlap = len(picked & set(hero_ids)) / len(hero_ids)

    context = {
        "scenes": N_SCENES, "variants": VARIANTS, "photos": len(rows),
        "near_dup_clusters": len(components), "shortlist": N_SCENES,
        "machine": platform.platform(),
    }
    benchmarks.record_benchmark(conn, name="triage_overlap", value=round(overlap, 4),
                                unit="fraction", n=len(hero_ids), phase="5", context=context)
    benchmarks.record_benchmark(conn, name="triage_throughput", value=round(throughput, 1),
                                unit="photos/s", n=len(rows), phase="5", context=context)
    conn.commit()
    conn.close()

    print("\n=== Phase 5 triage benchmark ===")
    print(f"  photos:              {len(rows)} ({N_SCENES} heroes x {VARIANTS} variants)")
    print(f"  near-dup clusters:   {len(components)}")
    print(f"  shortlist size:      {N_SCENES}")
    print(f"  triage_overlap:      {overlap:.3f}  ({len(picked & set(hero_ids))}/{len(hero_ids)} heroes kept)")
    print(f"  triage_throughput:   {throughput:.1f} photos/s")
    print(f"\n  logged 2 rows to benchmarks at {work / 'bench.sqlite'}")


if __name__ == "__main__":
    main()
