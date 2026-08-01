"""OCR service: owns the (lazily loaded) recognition engine (ARCHITECTURE §6)."""

from __future__ import annotations

import logging

from iris.config import Settings
from iris.ocr.engine import OcrEngine, OcrRegion, load_engine

logger = logging.getLogger("iris.ocr")


class OcrService:
    def __init__(self, settings: Settings, *, engine: OcrEngine | None = None) -> None:
        self._settings = settings
        self._engine = engine
        self._loaded = engine is not None

    def engine(self) -> OcrEngine | None:
        """Load the configured engine once; None means OCR is unavailable on this host."""
        if not self._loaded:
            self._engine = load_engine(
                self._settings.ocr_engine,
                languages=self._settings.ocr_languages,
                min_conf=self._settings.ocr_min_conf,
                max_edge=self._settings.ocr_max_edge,
            )
            self._loaded = True
        return self._engine

    @staticmethod
    def combined_text(regions: list[OcrRegion]) -> str:
        """Newline-join recognized lines into the blob stored in the FTS index."""
        return "\n".join(r.text for r in regions)
