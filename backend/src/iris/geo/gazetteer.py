"""Offline gazetteer: place name ↔ coordinates (ARCHITECTURE §4).

Forward geocoding ("Paris" → 48.85, 2.35) and reverse ("what city is this GPS near?")
run **entirely offline** against ``geonamescache``'s bundled ~34k-city dataset — no
network, no API key, consistent with the local-first constraint. The heavy import + index
build are lazy; when the ``geo`` extra isn't installed, ``load_gazetteer`` returns None and
place search degrades to "unavailable" (GPS-coordinate filters still work without it).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("iris.geo")

_EARTH_KM = 6371.0


@dataclass(frozen=True)
class GeoPlace:
    name: str
    country: str
    lat: float
    lon: float
    population: int

    @property
    def label(self) -> str:
        return f"{self.name}, {self.country}" if self.country else self.name


def bbox_for(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Lat/lon bounding box enclosing a radius around a point (SQL prefilter)."""
    dlat = math.degrees(radius_km / _EARTH_KM)
    cos_lat = max(0.01, math.cos(math.radians(lat)))
    dlon = math.degrees(radius_km / (_EARTH_KM * cos_lat))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


class Gazetteer:
    """In-memory name index + coordinate arrays for reverse lookup."""

    def __init__(self, places: list[GeoPlace], name_index: dict[str, list[int]]) -> None:
        self._places = places
        self._name_index = name_index
        self._lats = np.array([p.lat for p in places], dtype=np.float64)
        self._lons = np.array([p.lon for p in places], dtype=np.float64)

    def geocode(self, name: str) -> GeoPlace | None:
        """Resolve a place name to its most prominent (highest-population) match.

        Disambiguation is by population, so "Paris" yields Paris, FR over Paris, TX.
        """
        key = name.strip().lower()
        if not key:
            return None
        # allow "paris, fr" / "paris, france" by falling back to the leading token
        candidates = self._name_index.get(key)
        if candidates is None and "," in key:
            candidates = self._name_index.get(key.split(",", 1)[0].strip())
        if not candidates:
            return None
        return max((self._places[i] for i in candidates), key=lambda p: p.population)

    def reverse(self, lat: float, lon: float) -> tuple[GeoPlace, float] | None:
        """Nearest city to a coordinate, with its great-circle distance in km."""
        if not self._places:
            return None
        dists = _haversine_vec(lat, lon, self._lats, self._lons)
        idx = int(np.argmin(dists))
        return self._places[idx], float(dists[idx])


def _haversine_vec(
    lat: float, lon: float, lats: npt.NDArray[np.float64], lons: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    p1 = math.radians(lat)
    p2 = np.radians(lats)
    dphi = np.radians(lats - lat)
    dlambda = np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    dist: npt.NDArray[np.float64] = 2 * _EARTH_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return dist


def load_gazetteer(min_population: int = 0) -> Gazetteer | None:
    """Build the gazetteer from geonamescache, or None if the ``geo`` extra is missing."""
    try:
        import geonamescache
    except Exception:
        logger.info("geonamescache not installed; place-name search unavailable")
        return None

    cities = geonamescache.GeonamesCache().get_cities()
    places: list[GeoPlace] = []
    name_index: dict[str, list[int]] = {}
    for record in cities.values():
        try:
            population = int(record.get("population") or 0)
        except (TypeError, ValueError):
            population = 0
        if population < min_population:
            continue
        place = GeoPlace(
            name=record["name"],
            country=record.get("countrycode", ""),
            lat=float(record["latitude"]),
            lon=float(record["longitude"]),
            population=population,
        )
        idx = len(places)
        places.append(place)
        keys = {place.name.lower()}
        alt = record.get("alternatenames") or []
        tokens = alt.split(",") if isinstance(alt, str) else alt  # shape varies by version
        for raw in tokens:
            token = str(raw).strip().lower()
            if token and token.isascii():
                keys.add(token)
        for key in keys:
            name_index.setdefault(key, []).append(idx)

    logger.info("gazetteer loaded: %d cities, %d name keys", len(places), len(name_index))
    return Gazetteer(places, name_index)
