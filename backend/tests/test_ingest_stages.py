"""Unit tests for the pure ingest stage functions."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from iris.ingest.metadata import hash_file, parse_exif_datetime, read_metadata
from iris.ingest.phash import hamming_distance, perceptual_hash
from iris.ingest.thumbnails import render_thumbnail
from iris.storage import content_shard_path


def _make_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 48)) -> None:
    Image.new("RGB", size, color).save(path)


def test_content_shard_path() -> None:
    p = content_shard_path(Path("/thumbs"), "abcdef123456", "webp")
    assert p == Path("/thumbs/ab/cd/abcdef123456.webp")


def test_hash_file_is_stable_and_content_addressed(tmp_path: Path) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello world")
    assert hash_file(a) == hash_file(b)  # same content -> same digest
    b.write_bytes(b"different")
    assert hash_file(a) != hash_file(b)


def test_parse_exif_datetime() -> None:
    assert parse_exif_datetime("2021:07:04 12:30:00") is not None
    assert parse_exif_datetime("not a date") is None
    assert parse_exif_datetime("") is None


def test_read_metadata_dimensions(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    _make_image(img, (10, 20, 30), size=(100, 50))
    meta = read_metadata(img)
    assert meta.width == 100
    assert meta.height == 50


def test_read_metadata_unreadable_is_graceful(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    meta = read_metadata(bad)  # must not raise
    assert meta.width is None


def test_perceptual_hash_stable_and_discriminating(tmp_path: Path) -> None:
    solid = Image.new("RGB", (200, 200), (120, 120, 120))
    assert perceptual_hash(solid) == perceptual_hash(solid.copy())
    # A structured image should differ from a flat one.
    striped = Image.new("RGB", (200, 200), (0, 0, 0))
    for x in range(0, 200, 8):
        for y in range(200):
            striped.putpixel((x, y), (255, 255, 255))
    assert hamming_distance(perceptual_hash(solid), perceptual_hash(striped)) > 0


def test_render_thumbnail_writes_sharded_webp(tmp_path: Path) -> None:
    src = tmp_path / "src.png"
    _make_image(src, (200, 100, 50), size=(800, 600))
    thumbs = tmp_path / "thumbs"
    digest = "deadbeefcafe0001"
    result = render_thumbnail(1, str(src), digest, str(thumbs))
    assert result.ok
    assert result.width == 800 and result.height == 600  # true (pre-thumb) dims
    assert result.phash is not None
    out = content_shard_path(thumbs, digest, "webp")
    assert out.exists()
    with Image.open(out) as thumb:
        assert max(thumb.size) <= 256  # downscaled to the max edge


def test_render_thumbnail_handles_heic(tmp_path: Path) -> None:
    import pillow_heif

    pillow_heif.register_heif_opener()
    src = tmp_path / "photo.heic"
    Image.new("RGB", (600, 400), (90, 140, 200)).save(src, format="HEIF", quality=70)
    result = render_thumbnail(3, str(src), "heic000000000001", str(tmp_path / "thumbs"))
    assert result.ok
    assert result.width == 600 and result.height == 400
    assert content_shard_path(tmp_path / "thumbs", "heic000000000001", "webp").exists()


def test_render_thumbnail_bad_file_is_not_fatal(tmp_path: Path) -> None:
    bad = tmp_path / "bad.heic"
    bad.write_bytes(b"garbage")
    result = render_thumbnail(2, str(bad), "0" * 16, str(tmp_path / "thumbs"))
    assert not result.ok
    assert result.error is not None
