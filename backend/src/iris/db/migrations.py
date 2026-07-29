"""Forward-only SQL migration runner.

Migrations are ``NNNN_name.sql`` files in ``migrations/``. The applied version is
stored in ``meta['schema_version']``. On startup we apply every migration whose
numeric prefix is greater than the stored version, each in its own transaction.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _discover() -> list[tuple[int, Path]]:
    """Return ``(version, path)`` for every migration file, sorted ascending."""
    found: list[tuple[int, Path]] = []
    for path in MIGRATIONS_DIR.glob("*.sql"):
        prefix = path.name.split("_", 1)[0]
        found.append((int(prefix), path))
    found.sort(key=lambda item: item[0])
    return found


def current_version(conn: sqlite3.Connection) -> int:
    """Applied schema version, or 0 if the database is uninitialized."""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.OperationalError:
        return 0  # meta table does not exist yet
    return int(row[0]) if row is not None else 0


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Apply all pending migrations. Returns the resulting schema version.

    Each migration runs atomically: the DDL and the ``schema_version`` bump are
    wrapped in a single ``BEGIN ... COMMIT`` executed via ``executescript`` (the
    version is an integer parsed from the filename, so inlining it is injection-safe).
    """
    version = current_version(conn)
    for target, path in _discover():
        if target <= version:
            continue
        sql = path.read_text(encoding="utf-8")
        script = (
            "BEGIN;\n"
            f"{sql}\n"
            "INSERT INTO meta (key, value) "
            f"VALUES ('schema_version', '{target}') "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value;\n"
            "COMMIT;"
        )
        try:
            conn.executescript(script)
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        version = target
    return version
