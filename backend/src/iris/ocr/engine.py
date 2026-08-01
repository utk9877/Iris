"""OCR engines (ARCHITECTURE §6/§10).

The default engine is **Apple's Vision framework** via ``ocrmac`` — native, on-device, and
fast on Apple Silicon — chosen over the originally-planned PaddleOCR whose ``paddlepaddle``
wheel is unreliable on ARM Macs (CLAUDE.md). The heavy import is lazy and macOS-guarded, so
non-macOS hosts (CI/Linux) simply get no engine and the ingest OCR stage skips gracefully,
exactly like the faces stage. A different engine (e.g. Paddle) can implement ``OcrEngine``
and be selected via ``Settings.ocr_engine`` without touching the pipeline.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

logger = logging.getLogger("iris.ocr")


@dataclass
class OcrRegion:
    """One recognized text line with a normalized top-left-origin bbox and confidence."""

    bx: float
    by: float
    bw: float
    bh: float
    conf: float
    text: str


class OcrEngine(Protocol):
    """A text recognizer over an image file path."""

    def recognize(self, path: str) -> list[OcrRegion]: ...


def _load_scaled(path: str, max_edge: int) -> Image.Image:
    """Load an image RGB, downscaled so its longest edge is at most ``max_edge`` px."""
    image = Image.open(path).convert("RGB")
    long_edge = max(image.size)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    return image


class AppleVisionOCR:
    """Apple Vision recognizer (lazy ``ocrmac`` import; requires macOS + the ``ocr`` extra)."""

    def __init__(self, *, languages: Sequence[str], min_conf: float, max_edge: int) -> None:
        from ocrmac import ocrmac  # noqa: F401  (import-time availability check)

        self._ocrmac = ocrmac
        self._languages = list(languages)
        self._min_conf = min_conf
        self._max_edge = max_edge

    def recognize(self, path: str) -> list[OcrRegion]:
        image = _load_scaled(path, self._max_edge)
        annotations = self._ocrmac.OCR(
            image, recognition_level="accurate", language_preference=self._languages
        ).recognize()
        regions: list[OcrRegion] = []
        for text, confidence, bbox in annotations:
            if confidence < self._min_conf or not text.strip():
                continue
            x, y, w, h = bbox  # Vision: normalized, origin bottom-left
            regions.append(
                OcrRegion(bx=x, by=1.0 - (y + h), bw=w, bh=h, conf=float(confidence), text=text)
            )
        return regions


def load_engine(
    name: str, *, languages: Sequence[str], min_conf: float, max_edge: int
) -> OcrEngine | None:
    """Instantiate the configured OCR engine, or None if it is unavailable here.

    Returning None (rather than raising) lets the ingest stage skip OCR cleanly on hosts
    without a working engine — non-macOS, or macOS missing the ``ocr`` extra.
    """
    if name in ("none", ""):
        return None
    if name == "apple_vision":
        if sys.platform != "darwin":
            logger.info("Apple Vision OCR unavailable on %s; skipping OCR", sys.platform)
            return None
        try:
            return AppleVisionOCR(languages=languages, min_conf=min_conf, max_edge=max_edge)
        except Exception:
            logger.warning("failed to load Apple Vision OCR (is the 'ocr' extra installed?)")
            return None
    logger.warning("unknown OCR engine %r; skipping OCR", name)
    return None
