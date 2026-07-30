"""Light metadata stages: blake3 file hash and EXIF/dimension extraction.

These are IO-light and run in a thread pool (native hashing / header reads release
the GIL). EXIF parsing is best-effort: a photo with unreadable metadata still gets
a hash and is not treated as fatal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from blake3 import blake3
from PIL import Image

_HASH_CHUNK = 1 << 20  # 1 MiB

# EXIF tag ids (see CIPA DC-008 / Pillow ExifTags).
_TAG_MAKE = 271
_TAG_MODEL = 272
_TAG_ORIENTATION = 274
_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825
_TAG_DATETIME_ORIGINAL = 36867
_TAG_ISO = 34855
_TAG_FNUMBER = 33437
_TAG_EXPOSURE = 33434
_TAG_FOCAL = 37386
_TAG_LENS = 42036


def hash_file(path: Path) -> str:
    """Streaming blake3 hex digest of a file's bytes."""
    hasher = blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(slots=True)
class MetaResult:
    """Outcome of the hash + EXIF stage for one photo."""

    photo_id: int
    ok: bool
    content_hash: str | None = None
    meta: ImageMetadata | None = None
    error: str | None = None


def extract_metadata(photo_id: int, path: str) -> MetaResult:
    """Thread-pool worker: hash the file (required) and read EXIF (best-effort).

    A failure to even read the bytes (e.g. the file vanished) yields ``ok=False``
    so the orchestrator can flag the photo missing.
    """
    try:
        digest = hash_file(Path(path))
    except OSError as exc:
        return MetaResult(photo_id, ok=False, error=f"{type(exc).__name__}: {exc}")
    return MetaResult(photo_id, ok=True, content_hash=digest, meta=read_metadata(Path(path)))


@dataclass(slots=True)
class ImageMetadata:
    width: int | None = None
    height: int | None = None
    orientation: int | None = None
    taken_at: float | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    lens: str | None = None
    iso: int | None = None
    f_number: float | None = None
    exposure: float | None = None
    focal_length: float | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None


def parse_exif_datetime(value: str) -> float | None:
    """Parse an EXIF ``YYYY:MM:DD HH:MM:SS`` string to a Unix timestamp.

    EXIF timestamps are naive (no zone); we interpret them as UTC for a stable,
    monotonic sort key. ``tz_offset`` is left NULL until we have a real offset.
    """
    try:
        dt = datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None
    return dt.replace(tzinfo=UTC).timestamp()


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _to_int(value: Any) -> int | None:
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _gps_to_degrees(coord: Any, ref: Any) -> float | None:
    try:
        degrees = float(coord[0]) + float(coord[1]) / 60.0 + float(coord[2]) / 3600.0
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None
    if isinstance(ref, str) and ref.upper() in ("S", "W"):
        degrees = -degrees
    return degrees


def read_metadata(path: Path) -> ImageMetadata:
    """Best-effort image dimensions + EXIF. Never raises on a decodable header."""
    meta = ImageMetadata()
    try:
        with Image.open(path) as image:
            meta.width, meta.height = image.size
            exif = image.getexif()
    except Exception:  # unreadable header — leave everything None
        return meta

    meta.orientation = _to_int(exif.get(_TAG_ORIENTATION))
    meta.camera_make = _clean_str(exif.get(_TAG_MAKE))
    meta.camera_model = _clean_str(exif.get(_TAG_MODEL))

    exif_ifd = exif.get_ifd(_EXIF_IFD)
    if exif_ifd:
        dto = exif_ifd.get(_TAG_DATETIME_ORIGINAL)
        if isinstance(dto, str):
            meta.taken_at = parse_exif_datetime(dto)
        meta.iso = _to_int(exif_ifd.get(_TAG_ISO))
        meta.f_number = _to_float(exif_ifd.get(_TAG_FNUMBER))
        meta.exposure = _to_float(exif_ifd.get(_TAG_EXPOSURE))
        meta.focal_length = _to_float(exif_ifd.get(_TAG_FOCAL))
        meta.lens = _clean_str(exif_ifd.get(_TAG_LENS))

    gps_ifd = exif.get_ifd(_GPS_IFD)
    if gps_ifd:
        meta.gps_lat = _gps_to_degrees(gps_ifd.get(2), gps_ifd.get(1))
        meta.gps_lon = _gps_to_degrees(gps_ifd.get(4), gps_ifd.get(3))

    return meta


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().strip("\x00").strip()
    return cleaned or None
