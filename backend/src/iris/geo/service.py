"""Geo service: owns the lazily built offline gazetteer (ARCHITECTURE §4)."""

from __future__ import annotations

import logging

from iris.config import Settings
from iris.geo.gazetteer import Gazetteer, GeoPlace, load_gazetteer

logger = logging.getLogger("iris.geo")


class GeoService:
    def __init__(self, settings: Settings, *, gazetteer: Gazetteer | None = None) -> None:
        self._settings = settings
        self._gazetteer = gazetteer
        self._loaded = gazetteer is not None

    def gazetteer(self) -> Gazetteer | None:
        """Build the gazetteer once; None means place search is unavailable on this host."""
        if not self._loaded:
            logger.info("loading offline gazetteer (first use)")
            self._gazetteer = load_gazetteer(min_population=self._settings.geo_min_population)
            self._loaded = True
        return self._gazetteer

    def resolve_place(self, name: str) -> GeoPlace | None:
        gaz = self.gazetteer()
        return gaz.geocode(name) if gaz is not None else None
