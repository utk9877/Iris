"""Deterministic per-image aesthetic + technical-quality scorers (ARCHITECTURE §7).

**Divergence from the plan, documented here and in ARCHITECTURE §7.** The plan named
the *LAION aesthetic predictor* for the aesthetic score. That model is a small MLP
head trained on **OpenAI CLIP ViT-L/14** (768-d) image embeddings; Iris embeds with
**ViT-B/32** (512-d), so the pretrained head is dimensionally incompatible. Rather
than double the model weight and re-embed the whole library under a second CLIP just
for one score, Phase 5 ships a transparent, fully-offline, deterministic heuristic
computed from the thumbnail with **numpy + Pillow only** (no cv2, no model download —
so it also runs in the embed-only CI environment). The scorer is intentionally a pure
function so a learned LAION/L-14 head can drop in later behind the same interface.

Both scores are squashed to ``[0, 1]``. Absolute magnitudes are only loosely
calibrated: triage re-normalizes every score by **percentile rank within the current
scope** (§7), so what matters here is a stable *ordering* — sharper beats blurrier,
more colorful and balanced beats flat — not the exact value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image

# Squash scales — the value of the raw statistic at which the score reaches ~0.63
# (1 - 1/e). Picked to spread typical thumbnail statistics across (0, 1); see the
# note above on why exact calibration is unnecessary.
_SHARPNESS_SCALE = 150.0  # variance of the Laplacian on 0..255 luma
_COLORFULNESS_SCALE = 50.0  # Hasler-Süsstrunk colorfulness metric
_CONTRAST_SCALE = 64.0  # std-dev of luma (0..255)

_F = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ImageScores:
    """Aesthetic and technical-quality scores, each in ``[0, 1]``."""

    aesthetic: float
    quality: float


def _saturating(x: float, scale: float) -> float:
    """Map ``[0, inf)`` to ``[0, 1)`` with a smooth exponential knee."""
    return 1.0 - math.exp(-max(x, 0.0) / scale)


def _luma(rgb: _F) -> _F:
    """Rec. 601 luminance of an ``(H, W, 3)`` float array."""
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _laplacian_var(luma: _F) -> float:
    """Variance of the 4-neighbour Laplacian — the classic focus/sharpness measure."""
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return 0.0
    lap = (
        luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:] - 4.0 * luma[1:-1, 1:-1]
    )
    return float(lap.var())


def _colorfulness(rgb: _F) -> float:
    """Hasler & Süsstrunk (2003) colorfulness metric."""
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    std = math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2)
    mean = math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)
    return std + 0.3 * mean


def _saturation(rgb: _F) -> float:
    """Mean HSV-style saturation ``(max - min) / max`` in ``[0, 1]``."""
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 0)
    return float(sat.mean())


def score_array(rgb_uint8: npt.NDArray[np.uint8]) -> ImageScores:
    """Score an ``(H, W, 3)`` uint8 RGB image."""
    rgb = rgb_uint8.astype(np.float64)
    luma = _luma(rgb)

    # --- Technical quality: sharpness, exposure, contrast ---
    sharpness = _saturating(_laplacian_var(luma), _SHARPNESS_SCALE)
    clipped = float(np.mean(luma < 8.0) + np.mean(luma > 247.0))
    brightness_dev = abs(float(luma.mean()) / 255.0 - 0.5) * 2.0  # 0 = mid-grey, 1 = extreme
    exposure = max(0.0, 1.0 - clipped - 0.5 * brightness_dev)
    contrast = min(1.0, float(luma.std()) / _CONTRAST_SCALE)
    quality = 0.5 * sharpness + 0.3 * exposure + 0.2 * contrast

    # --- Aesthetic: colorfulness, saturation, tonal contrast ---
    colorfulness = _saturating(_colorfulness(rgb), _COLORFULNESS_SCALE)
    saturation = _saturation(rgb)
    aesthetic = 0.45 * colorfulness + 0.3 * saturation + 0.25 * contrast

    return ImageScores(
        aesthetic=round(min(1.0, max(0.0, aesthetic)), 6),
        quality=round(min(1.0, max(0.0, quality)), 6),
    )


def score_image(image: Image.Image) -> ImageScores:
    """Score a Pillow image (converted to RGB)."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return score_array(arr)
