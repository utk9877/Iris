"""Application configuration and on-disk layout.

Resolves the app-managed data directory (see ARCHITECTURE §3) and exposes typed
settings. All derived state lives under the data dir; source images are never
touched. Every value can be overridden with an ``IRIS_``-prefixed env var, e.g.
``IRIS_PORT=9000`` or ``IRIS_DATA_DIR=/tmp/iris-test``.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_dir() -> Path:
    """Platform-appropriate application data directory.

    macOS: ``~/Library/Application Support/Iris`` (the Tauri ``appDataDir``).
    Other platforms fall back to ``~/.local/share/Iris`` for dev use.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Iris"
    return Path.home() / ".local" / "share" / "Iris"


class Settings(BaseSettings):
    """Typed application settings, overridable via ``IRIS_*`` env vars."""

    model_config = SettingsConfigDict(env_prefix="IRIS_", env_file=None)

    # --- Sidecar HTTP server ---
    host: str = "127.0.0.1"
    port: int = 8756

    # --- Storage ---
    data_dir: Path = Field(default_factory=_default_data_dir)

    # --- Database ---
    # SQLite busy timeout (ms) — see ARCHITECTURE §1/§10 (single-writer + WAL).
    sqlite_busy_timeout_ms: int = 5000

    # --- Ingest (ARCHITECTURE §2) ---
    metadata_workers: int = 8  # thread pool for hash + EXIF (IO-light)
    decode_workers: int = 0  # spawn process pool for decode+thumb; 0 -> os.cpu_count()

    # --- Embeddings / semantic search (ARCHITECTURE §2/§4), Phase 2 ---
    embed_model: str = "Xenova/clip-vit-base-patch32"  # pre-exported CLIP ONNX (HF)
    embed_dim: int = 512
    embed_batch: int = 32
    search_brute_max: int = 2000  # tier-1 threshold (T_small)
    search_filtered_max: int = 50000  # tier-2 threshold (T_medium)

    # --- Faces (ARCHITECTURE §6), Phase 3 ---
    face_model: str = "buffalo_l"  # insightface pack: SCRFD det_10g + ArcFace w600k_r50
    face_dim: int = 512
    face_det_size: int = 640
    face_min_size: int = 32  # drop faces smaller than this (px)
    face_min_det_score: float = 0.5
    # Thresholds calibrated against ArcFace (buffalo_l): different people measured
    # <=0.21 cosine, while the SAME person across pose/angle often sits at 0.3-0.6.
    # 0.50 was far too strict (split people by angle); 0.35/0.42 keep a safe margin
    # over the ~0.21 "stranger" ceiling. Tune per-library and re-cluster.
    face_assign_threshold: float = 0.42  # cosine to a cluster centroid to join it
    face_edge_threshold: float = 0.35  # cosine kNN-graph edge for Chinese Whispers
    face_cluster_k: int = 20
    face_pool_threshold: int = 200  # re-cluster the pending pool once it exceeds this
    face_max_edge: int = 1280  # downscale originals to this longest edge before detection
    face_batch: int = 16

    # --- Cache limits (ARCHITECTURE §3), config knobs surfaced early ---
    thumb_max_gb: float = 8.0
    preview_cache_gb: float = 2.0

    # ------------------------------------------------------------------ paths
    @property
    def db_path(self) -> Path:
        return self.data_dir / "iris.sqlite"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def previews_dir(self) -> Path:
        return self.cache_dir / "previews"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def all_dirs(self) -> tuple[Path, ...]:
        """Every directory the app manages, in creation order."""
        return (
            self.data_dir,
            self.embeddings_dir,
            self.index_dir,
            self.thumbs_dir,
            self.cache_dir,
            self.previews_dir,
            self.models_dir,
            self.logs_dir,
        )

    def ensure_dirs(self) -> None:
        """Create the app data directory tree if missing (idempotent)."""
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)

    def resolved_decode_workers(self) -> int:
        """Decode process-pool size, resolving 0 to the CPU count (min 1)."""
        return self.decode_workers if self.decode_workers > 0 else (os.cpu_count() or 4)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
