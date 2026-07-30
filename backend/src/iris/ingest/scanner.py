"""Filesystem scan: walk source roots and yield image files (read-only)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

IMAGE_EXTS = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
)


def iter_image_files(root: Path) -> Iterator[Path]:
    """Yield image files under ``root``, skipping hidden files/directories.

    Never writes, moves, or follows the tree outside ``root`` (no symlink follow).
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield Path(dirpath) / name
