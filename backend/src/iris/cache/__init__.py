"""Derived-file cache management: previews + LRU eviction (ARCHITECTURE §3, Phase 6)."""

from __future__ import annotations

from iris.cache.previews import (
    dir_size_bytes,
    enforce_cache_cap,
    ensure_preview,
    preview_path,
    tree_size_bytes,
)

__all__ = [
    "dir_size_bytes",
    "enforce_cache_cap",
    "ensure_preview",
    "preview_path",
    "tree_size_bytes",
]
