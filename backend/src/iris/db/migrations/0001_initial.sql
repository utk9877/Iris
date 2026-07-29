-- Iris schema v1 — see docs/ARCHITECTURE.md §1 (source of truth).
-- Derived vectors do NOT live here; photos.embed_row indexes a memmap file.

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE photos (
  id            INTEGER PRIMARY KEY,
  path          TEXT NOT NULL UNIQUE,       -- absolute source path (never mutated)
  dir           TEXT NOT NULL,              -- parent dir (folder filter)
  filename      TEXT NOT NULL,
  ext           TEXT NOT NULL,
  size_bytes    INTEGER NOT NULL,
  mtime         REAL NOT NULL,
  content_hash  TEXT,                       -- blake3 of file bytes (identity/dedupe)
  width         INTEGER,
  height        INTEGER,
  orientation   INTEGER,                    -- EXIF 1..8
  taken_at      REAL,                       -- EXIF DateTimeOriginal (nullable)
  tz_offset     INTEGER,
  camera_make   TEXT,
  camera_model  TEXT,
  lens          TEXT,
  iso           INTEGER,
  f_number      REAL,
  exposure      REAL,
  focal_length  REAL,
  gps_lat       REAL,
  gps_lon       REAL,
  phash         INTEGER,                     -- 64-bit perceptual hash (near-dup)
  aesthetic     REAL,                        -- LAION score
  quality       REAL,                        -- technical (sharpness/exposure)
  embed_row     INTEGER,                     -- row index into clip memmap (nullable)
  -- per-stage completion markers (NULL = pending; power crash-resume):
  hashed_at     REAL, exif_at REAL, thumb_at REAL,
  embed_at      REAL, phash_at REAL, faces_at REAL, ocr_at REAL,
  missing       INTEGER NOT NULL DEFAULT 0,  -- source file gone (soft-delete)
  created_at    REAL NOT NULL,
  updated_at    REAL NOT NULL
);
CREATE INDEX ix_photos_taken  ON photos(taken_at)          WHERE missing = 0;
CREATE INDEX ix_photos_dir    ON photos(dir);
CREATE INDEX ix_photos_hash   ON photos(content_hash);
CREATE INDEX ix_photos_camera ON photos(camera_model);
CREATE INDEX ix_photos_gps    ON photos(gps_lat, gps_lon)  WHERE gps_lat IS NOT NULL;
CREATE INDEX ix_photos_need_embed ON photos(id) WHERE embed_at IS NULL;
CREATE INDEX ix_photos_need_faces ON photos(id) WHERE faces_at IS NULL;
CREATE INDEX ix_photos_need_ocr   ON photos(id) WHERE ocr_at   IS NULL;

CREATE TABLE clusters (                       -- person clusters (face pipeline output)
  id          INTEGER PRIMARY KEY,
  label       TEXT,                           -- user-assigned name (nullable)
  pinned      INTEGER NOT NULL DEFAULT 0,      -- labeled clusters survive re-clustering
  rep_face_id INTEGER,                         -- FK to faces(id); set after faces exist
  size        INTEGER NOT NULL DEFAULT 0,      -- denormalized count
  centroid    BLOB,                            -- 512-f32 running centroid (fast assign)
  updated_at  REAL NOT NULL
);
CREATE INDEX ix_clusters_label ON clusters(label);

CREATE TABLE faces (
  id          INTEGER PRIMARY KEY,
  photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  bx REAL, by REAL, bw REAL, bh REAL,          -- bbox
  det_score   REAL NOT NULL,
  landmarks   BLOB,                            -- 5-point, for alignment
  quality     REAL,                            -- blur/pose/size gate
  embed_row   INTEGER,                         -- row in embeddings/faces.f32 memmap
  cluster_id  INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
  assigned    INTEGER NOT NULL DEFAULT 0,      -- 0 = pending pool
  created_at  REAL NOT NULL
);
CREATE INDEX ix_faces_photo   ON faces(photo_id);
CREATE INDEX ix_faces_cluster ON faces(cluster_id);
CREATE INDEX ix_faces_pending ON faces(id) WHERE assigned = 0;

CREATE TABLE groups (                          -- photo-level groupings
  id           INTEGER PRIMARY KEY,
  kind         TEXT NOT NULL,                  -- 'near_dup'|'burst'|'event'|'semantic'
  key          TEXT,                           -- event/semantic bucket key
  rep_photo_id INTEGER REFERENCES photos(id),
  size         INTEGER NOT NULL DEFAULT 0,
  score        REAL,                           -- cohesion / confidence
  start_at     REAL, end_at REAL,              -- time span (burst/event)
  created_at   REAL NOT NULL
);
CREATE INDEX ix_groups_kind     ON groups(kind);
CREATE INDEX ix_groups_kind_key ON groups(kind, key);
CREATE INDEX ix_groups_time     ON groups(kind, start_at);

CREATE TABLE group_items (
  group_id  INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  photo_id  INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  rank      REAL,                              -- position in group (burst-best = 0)
  PRIMARY KEY (group_id, photo_id)
);
CREATE INDEX ix_group_items_photo ON group_items(photo_id, group_id);

-- OCR full-text. Contentless FTS5, one row per photo (concatenated text).
CREATE VIRTUAL TABLE ocr USING fts5(
  text, photo_id UNINDEXED,
  tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE ocr_regions (                     -- per-box detail for highlight/debug
  id INTEGER PRIMARY KEY,
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  bx REAL, by REAL, bw REAL, bh REAL, conf REAL, text TEXT
);
CREATE INDEX ix_ocr_regions_photo ON ocr_regions(photo_id);

CREATE TABLE tags (
  id   INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL DEFAULT 'user'            -- 'user' | 'auto'
);
CREATE TABLE photo_tags (
  photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  tag_id     INTEGER NOT NULL REFERENCES tags(id)  ON DELETE CASCADE,
  source     TEXT NOT NULL DEFAULT 'user',
  confidence REAL,
  PRIMARY KEY (photo_id, tag_id)
);
CREATE INDEX ix_photo_tags_tag ON photo_tags(tag_id, photo_id);

CREATE TABLE jobs (
  id         INTEGER PRIMARY KEY,
  kind       TEXT NOT NULL,                    -- 'scan'|'embed'|'faces'|'ocr'|'cluster'|...
  state      TEXT NOT NULL,                    -- 'queued'|'running'|'done'|'error'|'canceled'
  target     TEXT,
  total      INTEGER NOT NULL DEFAULT 0,
  done       INTEGER NOT NULL DEFAULT 0,
  errored    INTEGER NOT NULL DEFAULT 0,
  checkpoint TEXT,                             -- JSON resume cursor
  error_msg  TEXT,
  started_at REAL, updated_at REAL, finished_at REAL
);
CREATE INDEX ix_jobs_state ON jobs(kind, state);

CREATE TABLE benchmarks (                      -- required by CLAUDE.md standing instruction
  id         INTEGER PRIMARY KEY,
  job_id     INTEGER REFERENCES jobs(id),
  name       TEXT NOT NULL,                    -- 'embed_throughput','search_p50',...
  value      REAL NOT NULL,
  unit       TEXT NOT NULL,                    -- 'img/s','ms','recall@10',...
  n          INTEGER,                          -- dataset/sample size
  phase      TEXT, git_sha TEXT,
  context    TEXT,                             -- JSON: machine, EP, batch, model
  created_at REAL NOT NULL
);
CREATE INDEX ix_benchmarks_name ON benchmarks(name, created_at);
