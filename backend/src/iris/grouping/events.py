"""Time/GPS event segmentation and burst detection (ARCHITECTURE §5).

Both walk the timeline-sorted photo rows linearly. An **event** starts on a large time
gap or a GPS jump; a **burst** is a run of same-camera frames a couple of seconds apart
*within* an event. event ⊃ burst forms the temporal hierarchy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

Row = dict[str, Any]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def _gps_jump_km(prev: Row, cur: Row) -> float | None:
    """Distance between consecutive shots, or None if either lacks coordinates."""
    if None in (prev["gps_lat"], prev["gps_lon"], cur["gps_lat"], cur["gps_lon"]):
        return None
    return _haversine_km(prev["gps_lat"], prev["gps_lon"], cur["gps_lat"], cur["gps_lon"])


def segment_events(rows: Sequence[Row], *, gap_seconds: float, gps_km: float) -> list[list[Row]]:
    """Split timeline-ordered rows into events on a time gap or GPS jump."""
    events: list[list[Row]] = []
    current: list[Row] = []
    for row in rows:
        if not current:
            current = [row]
            continue
        prev = current[-1]
        time_gap = (row["sort_at"] or 0.0) - (prev["sort_at"] or 0.0)
        jump = _gps_jump_km(prev, row)
        if time_gap > gap_seconds or (jump is not None and jump > gps_km):
            events.append(current)
            current = [row]
        else:
            current.append(row)
    if current:
        events.append(current)
    return events


def detect_bursts(event_rows: Sequence[Row], *, gap_seconds: float) -> list[list[Row]]:
    """Runs of same-camera frames within ``gap_seconds`` of each other."""
    bursts: list[list[Row]] = []
    run: list[Row] = []
    for row in event_rows:
        if not run:
            run = [row]
            continue
        prev = run[-1]
        time_gap = (row["sort_at"] or 0.0) - (prev["sort_at"] or 0.0)
        same_camera = row["camera_model"] == prev["camera_model"]
        if time_gap <= gap_seconds and same_camera:
            run.append(row)
        else:
            if len(run) >= 2:
                bursts.append(run)
            run = [row]
    if len(run) >= 2:
        bursts.append(run)
    return bursts
