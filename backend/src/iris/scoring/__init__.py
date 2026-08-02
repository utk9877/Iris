"""Per-photo aesthetic + technical quality scoring (ARCHITECTURE §7, Phase 5)."""

from __future__ import annotations

from iris.scoring.quality import ImageScores, score_array, score_image

__all__ = ["ImageScores", "score_array", "score_image"]
