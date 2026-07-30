-- Phase 1 additions — see docs/ARCHITECTURE.md §1/§3/§9.

-- Library source folders to scan (non-destructive; read-only).
CREATE TABLE roots (
  id       INTEGER PRIMARY KEY,
  path     TEXT NOT NULL UNIQUE,   -- absolute source directory
  added_at REAL NOT NULL
);

-- Stable, non-null sort key for the timeline grid: prefer EXIF taken_at, fall
-- back to file mtime. Enables keyset pagination without NULL-handling (§9).
ALTER TABLE photos ADD COLUMN sort_at REAL;
CREATE INDEX ix_photos_sort ON photos(sort_at, id);
