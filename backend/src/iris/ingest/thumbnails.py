"""Decode + orient + thumbnail + perceptual hash.

This is the CPU-heavy stage; ``render_thumbnail`` is a top-level, picklable
function run in a spawn ``ProcessPoolExecutor`` (ARCHITECTURE §2) so a bad decode
is isolated from the server. It writes a content-addressed WebP thumbnail with an
atomic temp+rename, and returns the true (post-orientation) dimensions + phash.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pillow_heif
from PIL import Image, ImageOps

from iris.ingest.phash import perceptual_hash
from iris.storage import content_shard_path

# Register the HEIC/HEIF opener in every worker process (spawn re-imports modules).
pillow_heif.register_heif_opener()

THUMB_MAX_EDGE = 256
THUMB_QUALITY = 80


@dataclass(slots=True)
class ThumbResult:
    photo_id: int
    ok: bool
    width: int | None = None
    height: int | None = None
    phash: int | None = None
    error: str | None = None


def render_thumbnail(
    photo_id: int,
    src: str,
    content_hash: str,
    thumbs_dir: str,
    max_edge: int = THUMB_MAX_EDGE,
    quality: int = THUMB_QUALITY,
) -> ThumbResult:
    """Decode ``src``, write its WebP thumbnail, and return dims + phash."""
    try:
        with Image.open(src) as raw:
            image = ImageOps.exif_transpose(raw)  # bake in EXIF orientation
            image = image.convert("RGB")
            width, height = image.size
            ph = perceptual_hash(image)
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

            out_path = content_shard_path(Path(thumbs_dir), content_hash, "webp")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.parent / f".{content_hash}.{os.getpid()}.tmp"
            image.save(tmp_path, format="WEBP", quality=quality, method=4)
            os.replace(tmp_path, out_path)  # atomic within the same directory
    except Exception as exc:  # isolate bad files — never crash the pipeline
        return ThumbResult(photo_id, ok=False, error=f"{type(exc).__name__}: {exc}")
    return ThumbResult(photo_id, ok=True, width=width, height=height, phash=ph)
