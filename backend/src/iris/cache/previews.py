"""On-demand preview cache with LRU eviction (ARCHITECTURE §3, Phase 6).

Previews are larger renders (default 1024 px long edge) of the *original*, kept in
``cache/previews/<content_hash>.webp`` — content-addressed so identical files share one
preview and a source rename never orphans it. Unlike thumbnails (durable), previews are
a **hard-capped LRU cache** (``preview_cache_gb``): each access touches the file's mtime,
and after a render we evict the least-recently-used files until the directory is back
under the cap. Rendering is a read-only decode of the source — never mutates originals.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pillow_heif
from PIL import Image, ImageOps

logger = logging.getLogger("iris.cache")

pillow_heif.register_heif_opener()


def preview_path(previews_dir: Path, content_hash: str) -> Path:
    """Content-addressed preview location (flat, per ARCHITECTURE §3)."""
    return previews_dir / f"{content_hash}.webp"


def dir_size_bytes(directory: Path) -> int:
    """Total bytes of regular files directly in ``directory`` (previews are flat)."""
    total = 0
    if not directory.exists():
        return 0
    for entry in os.scandir(directory):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def tree_size_bytes(directory: Path) -> int:
    """Total bytes of all regular files under ``directory`` (recursive)."""
    total = 0
    if not directory.exists():
        return 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def enforce_cache_cap(previews_dir: Path, max_bytes: int, *, keep: Path | None = None) -> int:
    """Evict least-recently-used previews until the dir is within ``max_bytes``.

    Recency is the file mtime (bumped on every serve). ``keep`` is never evicted (the
    file just rendered/served). Returns the number of bytes reclaimed.
    """
    if not previews_dir.exists():
        return 0
    entries: list[tuple[float, int, Path]] = []
    total = 0
    for entry in os.scandir(previews_dir):
        if not entry.is_file():
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        total += stat.st_size
        entries.append((stat.st_mtime, stat.st_size, Path(entry.path)))

    if total <= max_bytes:
        return 0
    entries.sort(key=lambda e: e[0])  # oldest first
    reclaimed = 0
    keep_resolved = keep.resolve() if keep is not None else None
    for _mtime, size, path in entries:
        if total - reclaimed <= max_bytes:
            break
        if keep_resolved is not None and path.resolve() == keep_resolved:
            continue
        try:
            path.unlink()
            reclaimed += size
        except OSError:
            continue
    if reclaimed:
        logger.info("evicted %d bytes from preview cache", reclaimed)
    return reclaimed


def render_preview(src: Path, dest: Path, max_edge: int, quality: int) -> None:
    """Decode ``src``, orient it, and write a WebP preview via atomic temp+rename."""
    with Image.open(src) as raw:
        image = ImageOps.exif_transpose(raw)
        image = image.convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f".{dest.stem}.{os.getpid()}.tmp"
        image.save(tmp, format="WEBP", quality=quality, method=4)
        os.replace(tmp, dest)


def ensure_preview(
    previews_dir: Path,
    content_hash: str,
    src: Path,
    *,
    max_edge: int,
    quality: int,
    cap_bytes: int,
) -> Path:
    """Return a cached preview path, rendering it on demand and enforcing the LRU cap.

    A cache hit just touches the file's mtime (LRU bookkeeping). A miss renders from the
    source, then evicts oldest previews until the directory is within ``cap_bytes``.
    """
    dest = preview_path(previews_dir, content_hash)
    if dest.exists():
        now = None  # os.utime(None) -> set atime/mtime to current time
        try:
            os.utime(dest, now)
        except OSError:
            pass
        return dest
    render_preview(src, dest, max_edge, quality)
    enforce_cache_cap(previews_dir, cap_bytes, keep=dest)
    return dest
