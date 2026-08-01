"""Gazetteer geocode/reverse + bbox math (fake in-memory data — no geonamescache)."""

from __future__ import annotations

from iris.geo.gazetteer import Gazetteer, GeoPlace, bbox_for


def _gazetteer() -> Gazetteer:
    places = [
        GeoPlace("Paris", "FR", 48.8534, 2.3488, 2_138_551),
        GeoPlace("Paris", "US", 33.6609, -95.5555, 25_171),  # Paris, Texas
        GeoPlace("Tokyo", "JP", 35.6895, 139.6917, 8_336_599),
    ]
    name_index: dict[str, list[int]] = {}
    for i, p in enumerate(places):
        name_index.setdefault(p.name.lower(), []).append(i)
    return Gazetteer(places, name_index)


def test_geocode_disambiguates_by_population() -> None:
    place = _gazetteer().geocode("paris")
    assert place is not None and place.country == "FR"  # the big one wins


def test_geocode_unknown_returns_none() -> None:
    assert _gazetteer().geocode("atlantis") is None


def test_reverse_finds_nearest_city() -> None:
    result = _gazetteer().reverse(48.86, 2.35)
    assert result is not None
    place, dist_km = result
    assert place.country == "FR"
    assert dist_km < 5.0


def test_bbox_encloses_radius() -> None:
    min_lat, min_lon, max_lat, max_lon = bbox_for(48.8534, 2.3488, 25.0)
    assert min_lat < 48.8534 < max_lat
    assert min_lon < 2.3488 < max_lon
    # ~25 km is well under 1 degree of latitude (~111 km)
    assert (max_lat - min_lat) < 1.0
