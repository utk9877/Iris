"""Preview cache: render, LRU touch-on-hit, and cap eviction."""

from __future__ import annotations

import os
import time
from pathlib import Path

from PIL import Image

from iris.cache import previews


def _src(path: Path, size: tuple[int, int] = (2000, 1500)) -> Path:
    Image.new("RGB", size, (120, 60, 30)).save(path)
    return path


def test_render_preview_downscales(tmp_path: Path) -> None:
    src = _src(tmp_path / "orig.png")
    dest = tmp_path / "prev.webp"
    previews.render_preview(src, dest, max_edge=1024, quality=82)
    assert dest.exists()
    with Image.open(dest) as im:
        assert max(im.size) == 1024  # long edge clamped


def test_ensure_preview_hit_touches_mtime(tmp_path: Path) -> None:
    previews_dir = tmp_path / "previews"
    src = _src(tmp_path / "o.png")
    p1 = previews.ensure_preview(
        previews_dir, "abc", src, max_edge=512, quality=80, cap_bytes=10**9
    )
    old = os.stat(p1).st_mtime
    os.utime(p1, (old - 1000, old - 1000))  # backdate
    p2 = previews.ensure_preview(
        previews_dir, "abc", src, max_edge=512, quality=80, cap_bytes=10**9
    )
    assert p1 == p2
    assert os.stat(p2).st_mtime > old - 1000  # cache hit bumped recency


def test_enforce_cache_cap_evicts_oldest(tmp_path: Path) -> None:
    d = tmp_path / "previews"
    d.mkdir()
    sizes = {}
    for i in range(5):
        f = d / f"{i}.webp"
        f.write_bytes(b"x" * 1000)
        os.utime(f, (1000 + i, 1000 + i))  # ascending mtime: 0 oldest, 4 newest
        sizes[i] = f
    # Cap at 3000 bytes -> evict the two oldest (0,1) so 3 files (3000 B) remain.
    reclaimed = previews.enforce_cache_cap(d, max_bytes=3000)
    assert reclaimed == 2000
    remaining = {int(p.stem) for p in d.glob("*.webp")}
    assert remaining == {2, 3, 4}


def test_enforce_cache_cap_respects_keep(tmp_path: Path) -> None:
    d = tmp_path / "previews"
    d.mkdir()
    for i in range(3):
        f = d / f"{i}.webp"
        f.write_bytes(b"y" * 1000)
        os.utime(f, (1000 + i, 1000 + i))
    keep = d / "0.webp"  # the oldest, but pinned
    previews.enforce_cache_cap(d, max_bytes=1500, keep=keep)
    assert keep.exists()  # never evicted despite being LRU-oldest


def test_dir_and_tree_size(tmp_path: Path) -> None:
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "a.webp").write_bytes(b"z" * 100)
    assert previews.dir_size_bytes(flat) == 100
    nested = tmp_path / "nested" / "ab" / "cd"
    nested.mkdir(parents=True)
    (nested / "x").write_bytes(b"z" * 250)
    assert previews.tree_size_bytes(tmp_path / "nested") == 250
    assert previews.dir_size_bytes(tmp_path / "missing") == 0


def test_ensure_preview_evicts_on_write(tmp_path: Path) -> None:
    previews_dir = tmp_path / "previews"
    previews_dir.mkdir()
    # Pre-fill with an old bulky file that should be evicted by a fresh render.
    old = previews_dir / "old.webp"
    old.write_bytes(b"x" * 5000)
    os.utime(old, (1, 1))
    src = _src(tmp_path / "o.png", (1200, 900))
    time.sleep(0.01)
    dest = previews.ensure_preview(
        previews_dir, "new", src, max_edge=1024, quality=82, cap_bytes=4000
    )
    assert dest.exists()  # the freshly rendered preview survives
    assert not old.exists()  # the stale bulky file was evicted to meet the cap
