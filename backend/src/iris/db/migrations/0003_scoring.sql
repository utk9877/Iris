-- Phase 5 — trip triage & aesthetics (ARCHITECTURE §7).
--
-- The `aesthetic` and `quality` columns already exist on `photos` (schema v1); this
-- migration only adds the per-photo *scoring stage* completion marker, mirroring the
-- other `*_at` stage markers that drive crash-resume. A photo is "scored" iff
-- `scored_at` is set. The partial index backs the ingest resume query
-- (`WHERE scored_at IS NULL`).

ALTER TABLE photos ADD COLUMN scored_at REAL;

CREATE INDEX ix_photos_need_score ON photos(id) WHERE scored_at IS NULL;
