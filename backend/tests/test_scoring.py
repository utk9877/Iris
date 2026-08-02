"""Deterministic aesthetic/quality scorer tests (pure numpy/PIL — no model, no cv2)."""

from __future__ import annotations

import numpy as np

from iris.scoring import score_array, score_image


def _checker(size: int = 64, block: int = 4) -> np.ndarray:
    """High-frequency black/white checkerboard — sharp, high-contrast."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if ((x // block) + (y // block)) % 2 == 0:
                arr[y, x] = 255
    return arr


def _flat(value: int = 128) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


def _colorful() -> np.ndarray:
    """A saturated rainbow gradient — high colorfulness."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
    arr[:, :, 1] = np.linspace(255, 0, 64, dtype=np.uint8)[:, None]
    arr[:, :, 2] = 128
    return arr


def test_scores_are_bounded() -> None:
    for arr in (_checker(), _flat(), _colorful(), _flat(0), _flat(255)):
        s = score_array(arr)
        assert 0.0 <= s.aesthetic <= 1.0
        assert 0.0 <= s.quality <= 1.0


def test_sharp_beats_flat_on_quality() -> None:
    assert score_array(_checker()).quality > score_array(_flat()).quality


def test_colorful_beats_grey_on_aesthetic() -> None:
    color = _colorful()
    grey = np.repeat(color.mean(axis=2, keepdims=True), 3, axis=2).astype(np.uint8)
    assert score_array(color).aesthetic > score_array(grey).aesthetic


def test_deterministic() -> None:
    arr = _colorful()
    assert score_array(arr) == score_array(arr)


def test_score_image_matches_array() -> None:
    from PIL import Image

    arr = _colorful()
    assert score_image(Image.fromarray(arr)) == score_array(arr)
