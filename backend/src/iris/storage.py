"""On-disk storage helpers (content-addressed derived files). See ARCHITECTURE §3."""

from __future__ import annotations

from pathlib import Path


def content_shard_path(base: Path, digest: str, ext: str) -> Path:
    """Two-level sharded path for a content-addressed file.

    ``base/ab/cd/<digest>.<ext>`` — keeps any single directory small and keeps
    identical files (same digest) mapped to one artifact.
    """
    return base / digest[:2] / digest[2:4] / f"{digest}.{ext}"
