"""Phase 4 OCR benchmark — REAL Apple Vision recognition, logged to `benchmarks`.

Measures on this machine (macOS + the `ocr` extra required):
  * ocr_recall — render images with KNOWN text (PIL), run the REAL Apple Vision engine,
    and measure the fraction of ground-truth words correctly recognized. Ground truth is
    what we drew, so recall is an honest measurement, not an estimate.
  * ocr_throughput — images/second through the real recognizer.

Run:  cd backend && uv run --extra ocr python ../scripts/bench_ocr.py
Skips with a clear message on non-macOS or when the `ocr` extra is missing.
"""

from __future__ import annotations

import platform
import re
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

from iris import benchmarks
from iris.config import get_settings
from iris.db import apply_migrations, connect
from iris.ocr.engine import load_engine

PHRASES = [
    "Boarding pass gate 22",
    "Invoice total 1499 dollars",
    "Welcome to Yosemite National Park",
    "Espresso 3 Latte 4 Muffin 2",
    "Departures Platform 9 Track 15",
    "Speed limit 55 mph ahead",
]
_WORD = re.compile(r"\w+")


def _render(text: str, path: Path) -> None:
    img = Image.new("RGB", (900, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 60), text, fill=(0, 0, 0))  # default PIL bitmap font
    img.save(path)


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def main() -> None:
    settings = get_settings()
    engine = load_engine(
        settings.ocr_engine,
        languages=settings.ocr_languages,
        min_conf=settings.ocr_min_conf,
        max_edge=settings.ocr_max_edge,
    )
    if engine is None:
        print(
            f"OCR engine '{settings.ocr_engine}' unavailable on {sys.platform} "
            "(need macOS + `uv sync --extra ocr`). Skipping — no number recorded."
        )
        return

    work = Path(tempfile.mkdtemp(prefix="iris-ocrbench-"))
    total_truth = 0
    total_hit = 0
    ocr_secs = 0.0
    for i, phrase in enumerate(PHRASES):
        path = work / f"text{i}.png"
        _render(phrase, path)
        t0 = time.perf_counter()
        regions = engine.recognize(str(path))
        ocr_secs += time.perf_counter() - t0
        found = set()
        for r in regions:
            found |= _words(r.text)
        truth = _words(phrase)
        total_truth += len(truth)
        total_hit += len(truth & found)

    recall = total_hit / total_truth if total_truth else 0.0
    throughput = len(PHRASES) / ocr_secs

    context = {
        "engine": settings.ocr_engine, "images": len(PHRASES), "words": total_truth,
        "font": "PIL default bitmap", "machine": platform.platform(),
    }
    conn = connect(work / "bench.sqlite")
    apply_migrations(conn)
    benchmarks.record_benchmark(conn, name="ocr_recall", value=round(recall, 4),
                                unit="fraction", n=total_truth, phase="4", context=context)
    benchmarks.record_benchmark(conn, name="ocr_throughput", value=round(throughput, 2),
                                unit="img/s", n=len(PHRASES), phase="4", context=context)
    conn.commit()
    conn.close()

    print("\n=== Phase 4 OCR benchmark (Apple Vision) ===")
    print(f"  images:         {len(PHRASES)}  | ground-truth words: {total_truth}")
    print(f"  ocr_recall:     {recall:.3f}  ({total_hit}/{total_truth} words)")
    print(f"  ocr_throughput: {throughput:.2f} img/s")
    print(f"\n  logged 2 rows to benchmarks at {work / 'bench.sqlite'}")


if __name__ == "__main__":
    main()
