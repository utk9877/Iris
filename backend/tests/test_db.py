"""Migration runner + schema v1 tests."""

from __future__ import annotations

from iris.config import Settings
from iris.db import apply_migrations, connect, current_version, latest_version

EXPECTED_TABLES = {
    "meta",
    "photos",
    "faces",
    "clusters",
    "groups",
    "group_items",
    "ocr",
    "ocr_regions",
    "tags",
    "photo_tags",
    "jobs",
    "benchmarks",
}


def test_migrations_apply_and_set_version(settings: Settings) -> None:
    settings.ensure_dirs()
    conn = connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    try:
        assert current_version(conn) == 0
        version = apply_migrations(conn)
        assert version == latest_version()
        assert current_version(conn) == latest_version()
    finally:
        conn.close()


def test_migrations_are_idempotent(settings: Settings) -> None:
    settings.ensure_dirs()
    conn = connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    try:
        assert apply_migrations(conn) == latest_version()
        # Re-running applies nothing and does not error.
        assert apply_migrations(conn) == latest_version()
    finally:
        conn.close()


def test_schema_has_expected_tables(settings: Settings) -> None:
    settings.ensure_dirs()
    conn = connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    try:
        apply_migrations(conn)
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            ).fetchall()
        }
        assert EXPECTED_TABLES <= names
    finally:
        conn.close()


def test_wal_mode_enabled(settings: Settings) -> None:
    settings.ensure_dirs()
    conn = connect(settings.db_path, busy_timeout_ms=settings.sqlite_busy_timeout_ms)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()
