# Iris Architecture

On-device, non-destructive photo library. FastAPI sidecar (indexing + ML + search)
behind a Tauri/React shell. Source images are read-only; all derived state lives in
an app-managed data directory.

> All performance numbers in this doc are **targets/estimates from arithmetic**,
> not measurements. Real figures come from actual runs logged to `benchmarks`
> (per CLAUDE.md standing instruction).

## 1. Data model (SQLite, WAL)

Single-writer, many-reader. Derived vectors do **not** live in SQLite (see
Embeddings). Timestamps are Unix epoch `REAL`. Per-stage nullable `*_at` columns on
`photos` drive crash-resume (a stage is "done" iff its column is set).

```sql
-- key/value meta: schema_version, model ids, memmap dims/dtype
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- library source folders to scan (read-only; non-destructive). [migration 0002]
CREATE TABLE roots (
  id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, added_at REAL NOT NULL
);

CREATE TABLE photos (
  id            INTEGER PRIMARY KEY,
  path          TEXT NOT NULL UNIQUE,      -- absolute source path (never mutated)
  dir           TEXT NOT NULL,             -- parent dir (folder filter)
  filename      TEXT NOT NULL,
  ext           TEXT NOT NULL,
  size_bytes    INTEGER NOT NULL,
  mtime         REAL NOT NULL,
  content_hash  TEXT,                      -- blake3 of file bytes (identity/dedupe)
  width         INTEGER,
  height        INTEGER,
  orientation   INTEGER,                   -- EXIF 1..8
  taken_at      REAL,                      -- EXIF DateTimeOriginal (nullable)
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
  phash         INTEGER,                   -- 64-bit perceptual hash (near-dup)
  sort_at       REAL,                      -- COALESCE(taken_at, mtime); keyset sort key [migration 0002]
  aesthetic     REAL,                      -- LAION score
  quality       REAL,                      -- technical (sharpness/exposure)
  embed_row     INTEGER,                   -- row index into clip memmap (nullable)
  -- per-stage completion markers (NULL = pending; power crash-resume):
  hashed_at     REAL, exif_at REAL, thumb_at REAL,
  embed_at      REAL, phash_at REAL, faces_at REAL, ocr_at REAL,
  missing       INTEGER NOT NULL DEFAULT 0,-- source file gone (soft-delete)
  created_at    REAL NOT NULL,
  updated_at    REAL NOT NULL
);
CREATE INDEX ix_photos_taken       ON photos(taken_at)           WHERE missing=0; -- timeline grid + keyset paging
CREATE INDEX ix_photos_sort        ON photos(sort_at, id);       -- keyset grid pagination [migration 0002]
CREATE INDEX ix_photos_dir         ON photos(dir);               -- folder filter
CREATE INDEX ix_photos_hash        ON photos(content_hash);      -- exact-dup / re-scan identity
CREATE INDEX ix_photos_camera      ON photos(camera_model);      -- camera filter
CREATE INDEX ix_photos_gps         ON photos(gps_lat, gps_lon)   WHERE gps_lat IS NOT NULL; -- map/event
-- resume scans (one partial index per stage; example shown):
CREATE INDEX ix_photos_need_embed  ON photos(id) WHERE embed_at IS NULL;
CREATE INDEX ix_photos_need_faces  ON photos(id) WHERE faces_at IS NULL;
CREATE INDEX ix_photos_need_ocr    ON photos(id) WHERE ocr_at   IS NULL;

-- Embeddings: NOT stored here. photos.embed_row -> row in embeddings/clip.f32 memmap.
-- Why: 100k x 768 f32 ~= 300 MB; as SQLite BLOBs it bloats the DB, evicts the page
-- cache, and cannot be bulk-scanned as a contiguous matrix. A memmap gives zero-copy
-- numpy views for brute-force matmul and feeds hnswlib directly. A small sidecar
-- meta file (dim, count, dtype, model id) tracks layout; free rows are compacted lazily.

CREATE TABLE faces (
  id          INTEGER PRIMARY KEY,
  photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  bx REAL, by REAL, bw REAL, bh REAL,      -- bbox
  det_score   REAL NOT NULL,
  landmarks   BLOB,                        -- 5-point, for alignment
  quality     REAL,                        -- blur/pose/size gate
  embed_row   INTEGER,                     -- row in embeddings/faces.f32 memmap
  cluster_id  INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
  assigned    INTEGER NOT NULL DEFAULT 0,  -- 0 = pending pool
  created_at  REAL NOT NULL
);
CREATE INDEX ix_faces_photo   ON faces(photo_id);
CREATE INDEX ix_faces_cluster ON faces(cluster_id);
CREATE INDEX ix_faces_pending ON faces(id) WHERE assigned=0; -- incremental pending pool

CREATE TABLE clusters (                    -- person clusters (face pipeline output)
  id          INTEGER PRIMARY KEY,
  label       TEXT,                        -- user-assigned name (nullable)
  pinned      INTEGER NOT NULL DEFAULT 0,  -- labeled clusters survive re-clustering
  rep_face_id INTEGER REFERENCES faces(id),
  size        INTEGER NOT NULL DEFAULT 0,  -- denormalized count
  centroid    BLOB,                        -- 512-f32 running centroid (fast assign)
  updated_at  REAL NOT NULL
);
CREATE INDEX ix_clusters_label ON clusters(label);

CREATE TABLE groups (                      -- photo-level groupings
  id          INTEGER PRIMARY KEY,
  kind        TEXT NOT NULL,               -- 'near_dup'|'burst'|'event'|'semantic'
  key         TEXT,                        -- event/semantic bucket key
  rep_photo_id INTEGER REFERENCES photos(id),
  size        INTEGER NOT NULL DEFAULT 0,
  score       REAL,                        -- cohesion / confidence
  start_at    REAL, end_at REAL,           -- time span (burst/event)
  created_at  REAL NOT NULL
);
CREATE INDEX ix_groups_kind     ON groups(kind);
CREATE INDEX ix_groups_kind_key ON groups(kind, key);
CREATE INDEX ix_groups_time     ON groups(kind, start_at);

CREATE TABLE group_items (
  group_id  INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  photo_id  INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  rank      REAL,                          -- position in group (burst-best = 0)
  PRIMARY KEY (group_id, photo_id)
);
CREATE INDEX ix_group_items_photo ON group_items(photo_id, group_id); -- "groups for photo"

-- OCR full-text. Contentless FTS5, one row per photo (concatenated text).
CREATE VIRTUAL TABLE ocr USING fts5(
  text, photo_id UNINDEXED,
  tokenize = 'unicode61 remove_diacritics 2'
);                                         -- query: WHERE ocr MATCH ?  (bm25 rank)
CREATE TABLE ocr_regions (                 -- per-box detail for highlight/debug
  id INTEGER PRIMARY KEY,
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  bx REAL, by REAL, bw REAL, bh REAL, conf REAL, text TEXT
);
CREATE INDEX ix_ocr_regions_photo ON ocr_regions(photo_id);

CREATE TABLE tags (
  id   INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL DEFAULT 'user'        -- 'user' | 'auto'
);
CREATE TABLE photo_tags (
  photo_id   INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  tag_id     INTEGER NOT NULL REFERENCES tags(id)  ON DELETE CASCADE,
  source     TEXT NOT NULL DEFAULT 'user',
  confidence REAL,
  PRIMARY KEY (photo_id, tag_id)
);
CREATE INDEX ix_photo_tags_tag ON photo_tags(tag_id, photo_id); -- "photos with tag"

CREATE TABLE jobs (
  id         INTEGER PRIMARY KEY,
  kind       TEXT NOT NULL,                -- 'scan'|'embed'|'faces'|'ocr'|'cluster'|...
  state      TEXT NOT NULL,                -- 'queued'|'running'|'done'|'error'|'canceled'
  target     TEXT,
  total      INTEGER NOT NULL DEFAULT 0,
  done       INTEGER NOT NULL DEFAULT 0,
  errored    INTEGER NOT NULL DEFAULT 0,
  checkpoint TEXT,                         -- JSON resume cursor
  error_msg  TEXT,
  started_at REAL, updated_at REAL, finished_at REAL
);
CREATE INDEX ix_jobs_state ON jobs(kind, state);

CREATE TABLE benchmarks (                  -- required by standing instruction
  id         INTEGER PRIMARY KEY,
  job_id     INTEGER REFERENCES jobs(id),
  name       TEXT NOT NULL,                -- 'embed_throughput','search_p50',...
  value      REAL NOT NULL,
  unit       TEXT NOT NULL,                -- 'img/s','ms','recall@10',...
  n          INTEGER,                      -- dataset/sample size
  phase      TEXT, git_sha TEXT,
  context    TEXT,                         -- JSON: machine, EP, batch, model
  created_at REAL NOT NULL
);
CREATE INDEX ix_benchmarks_name ON benchmarks(name, created_at);
```

## 2. Ingest pipeline

**Stage sequence:** `scan → hash → exif → decode+thumb → {embed, phash, faces, ocr}`.
Only decode is a hard prerequisite for embed/phash/faces/ocr; hash and exif are
independent of decode and of each other.

**Parallel vs serial / process boundary:**
- **Main process** — FastAPI + orchestrator + **single SQLite writer** (all writes
  funnel through one connection + a queue; WAL gives concurrent readers). Light,
  IO-bound stages (**scan** serial walker; **hash** and **exif** in a thread pool —
  native blake3 / header reads release the GIL) run here.
- **Decode is the fork point → `multiprocessing` (spawn) process pool**, sized to
  P-cores. Reasons: (a) full-image HEIC/JPEG decode is the CPU bottleneck and needs
  true parallelism past the GIL; (b) native decoders (libheif) can segfault on bad
  files — process isolation keeps one bad HEIC from killing the server. Each decode
  worker emits: the content-addressed **thumbnail** (to disk), a **CLIP-normalized
  tensor** (via `shared_memory`, no giant pickles), a **medium image for face
  detection**, and the **phash** (computed here, piggybacking the decode).
- **Inference workers are separate processes**, one ONNX Runtime session each
  (CoreML EP): an **embed** worker and a **faces** worker, fed by bounded queues and
  running **batched-serial** (the accelerator is the scarce resource; batching, not
  threading, is the win). Overlaps CPU decode.
- **OCR** is a decoupled, lowest-priority **process pool** (PaddleOCR) that may lag
  behind the rest; it reads the medium image and writes `ocr` + `ocr_regions`.

Bounded queues between stages provide backpressure. Pipeline parallelism: different
photos occupy different stages at once.

**Crash-resume:** each stage commits its output **and** its `photos.<stage>_at`
marker in one transaction; memmap row is written+flushed before the DB commit, so a
crash between them just re-does that photo next run (old memmap slot leaked, later
compacted — harmless). On startup the orchestrator re-enqueues, per stage,
`SELECT id FROM photos WHERE <stage>_at IS NULL AND <prereq>_at IS NOT NULL` (backed
by the partial indexes). The hnswlib index is a rebuildable cache — source of truth
is memmap + DB; on crash it is rebuilt/appended from rows with `embed_at` set. `jobs`
holds coarse progress + scan cursor.

## 3. Storage layout

App data dir (macOS `appDataDir`): `~/Library/Application Support/Iris/`.

```
Iris/
  iris.sqlite (+ -wal, -shm)
  embeddings/ clip.f32  clip.meta.json    # memmap N x 768 f32  (~300 MB @ 100k)
              faces.f32 faces.meta.json   # memmap M x 512 f32
  index/      clip.hnsw  faces.hnsw       # persisted hnswlib (rebuildable cache)
  thumbs/     ab/cd/<hash>.webp           # content-addressed, 2-level shard
  cache/      previews/<hash>.webp        # larger previews, LRU-capped
  models/     *.onnx                       # downloaded/converted on first run
  logs/
```

**Content-addressing:** thumbs/previews keyed by `photos.content_hash` (blake3),
path `thumbs/<h[0:2]>/<h[2:4]>/<h>.webp`. Benefits: identical files share one thumb;
source renames don't orphan (hash is content, not path); atomic writes via
temp-file + rename. Thumb = webp q≈80, 256 px long edge (~30 KB); preview = 1024 px.

**Cache limits (config knobs, defaults):** `thumbs/` is durable derived data (100k ×
~30 KB ≈ **~3 GB** arithmetic), soft cap `thumb_max_gb=8`. `cache/previews/` is
LRU-evicted, `preview_cache_gb=2`. Memmaps + DB are not capped (grow with library).

## 4. Search architecture

Text query → CLIP/SigLIP **text embedding** (query vector). Metadata filters (date,
folder, camera, GPS, person, tag, `has_text`) compile to **one SQL query** yielding a
candidate id set of size `S`. OCR full-text is another filter/source via `ocr MATCH`.

**Three-tier vector strategy** (starting thresholds `T_small=2,000`, `T_medium=50,000`
— to be tuned + logged):
- **Tier 1 — brute force** (`S ≤ T_small`): gather those rows from the clip memmap,
  exact cosine via a single numpy matmul, top-k. Exact and fastest at small N.
- **Tier 2 — filtered HNSW** (`T_small < S ≤ T_medium`): hnswlib search with an
  allow-list `filter` callable restricted to the candidate ids. Best when the filter
  is moderately selective.
- **Tier 3 — plain HNSW** (unfiltered, or `S > T_medium` i.e. filter barely dents
  100k): search the full index with over-fetch (`k' = 5×k`, `ef=128` start), then
  post-filter results against the metadata predicate.

**Fusion:** when both semantic and FTS are present, combine per-source ranks with
**Reciprocal Rank Fusion** (`k=60`); text-only queries rank by bm25; semantic-only by
cosine. Final sort selectable (relevance vs. `taken_at`). Results → hydrate `photos`.

## 5. Grouping architecture

All four are `groups` rows (+ `group_items`). Computed in a hierarchy so triage
dedupes before it diversifies:

1. **event** — time-gap segmentation over the `taken_at` timeline (+ GPS): new event
   when gap `> 4h` **or** GPS jump beyond threshold (start). Coarsest bucket.
2. **burst** — within an event, consecutive same-camera shots with inter-frame gap
   `< 2s` (start) → burst; `group_items.rank` orders by quality (best = 0).
3. **near_dup** — `phash` Hamming `≤ 6` bits (start) via sort+union-find, confirmed by
   CLIP cosine `≥ 0.95`. Catches near-identical frames across/within bursts.
4. **semantic** — global CLIP-embedding clustering (HDBSCAN over the kNN graph) into
   cross-cutting themes ("beach", "documents"); independent of time.

**Relationships:** event ⊃ burst ⊃ near_dup form a temporal hierarchy; semantic
cross-cuts them. **Ordering matters for triage:** near-dup collapse + burst-best
selection run **before** diversity ranking, so triage never diversifies over ten
identical frames. Semantic membership additionally feeds MMR diversity.

## 6. Face pipeline

`detect → align → embed → cluster → incremental-assign`.
- **Detect** — SCRFD (ONNX) on the medium image → bbox + 5 landmarks + `det_score`;
  drop faces `< 32 px` or `det_score < 0.5`.
- **Align** — similarity transform from the 5 landmarks to the canonical ArcFace
  112×112 template.
- **Embed** — ArcFace (ONNX) → 512-d, L2-normalized → `faces.f32` memmap. A quality
  score (Laplacian-variance blur, size, frontalness) gates clustering seeds.
- **Cluster** — build a kNN graph over face embeddings using the faces HNSW index
  (`k=20`, edge kept at cosine `≥ 0.5`), then **Chinese Whispers** label propagation
  over the graph → person `clusters` (near-linear, robust to cluster-count unknown).
  Rep face = highest-quality frontal.
- **Incremental / pending pool** — new faces are assigned to the nearest cluster
  `centroid` if cosine `≥ 0.60` (start): assign, bump centroid. Otherwise → **pending
  pool** (`assigned=0`). When the pool exceeds `200` (or on demand) re-run Chinese
  Whispers over *pending faces + cluster reps* to form/merge clusters. `pinned`
  (labeled) clusters keep their identity across re-clustering; user merge/split is
  respected.

## 7. Trip triage

Per-photo **four scores**: **technical quality** (OpenCV sharpness/exposure),
**aesthetic** (LAION predictor), **representativeness** (cosine to the event/semantic
centroid), **subject/face** (face presence × quality × known-person weight).

**Normalization:** **percentile-rank within the current scope** (event/trip) → [0,1].
Rank normalization is outlier-robust and makes the four comparable before weighting.

**Combination + diversify:** weighted sum of the normalized scores per preset, then
**MMR** over CLIP embeddings on the shortlist:
`pick argmax[ λ·score(i) − (1−λ)·max_{j∈sel} cos(i,j) ]`. MMR runs **after** near-dup
collapse + burst-best selection (§5).

**Preset profiles** `[quality, aesthetic, repr, subject]`, `λ`:
- **story-ready** — `[.25,.25,.30,.20]`, `λ=0.6` (balanced, more diverse).
- **print-worthy** — `[.35,.45,.10,.10]`, `λ=0.85` (favor beauty+technical; allow
  similar if both excellent).
- **delete-candidates** — *inverted*: rank by low quality + redundancy (near-dup group
  size, blur, closed eyes); skip MMR, instead surface all-but-best of each near-dup /
  burst group. Weights `[.5,.2,—,—]` on inverted quality/aesthetic + redundancy bonus.

## 8. API surface (FastAPI, localhost)

**Library / ingest:** `POST /library/roots{path}`, `GET/DELETE /library/roots`,
`POST /ingest/scan{root?}`, `GET /ingest/status`, `POST /ingest/cancel`,
`GET /jobs`, `GET /jobs/{id}`.
**Photos / media:** `GET /photos?cursor&limit&sort&filters…` → `{items:[{id,hash,
taken_at,w,h}], next_cursor}`; `GET /photos/{id}` (full meta); `GET /photos/count`;
`GET /thumb/{id}`, `GET /preview/{id}`, `GET /file/{id}` (read-only original stream).
**Search:** `POST /search{query?,filters,sort,cursor,limit}` → `{items:[{id,score}],
tier,next_cursor}`; `GET /search/suggest?q=`.
**Groups:** `GET /groups?kind&cursor`, `GET /groups/{id}` (ranked items).
**People / faces:** `GET /people`, `GET /people/{id}`, `PATCH /people/{id}{label}`,
`POST /people/merge{ids}`, `POST /people/{id}/split`, `GET /photos/{id}/faces`,
`POST /faces/recluster`.
**Text / tags:** `GET /photos/{id}/ocr`, `GET /tags`, `POST /photos/{id}/tags{name}`,
`DELETE /photos/{id}/tags/{tag_id}`.
**Triage:** `POST /triage{scope:event_id|group_id|date_range, preset, limit}` →
`{items:[{id,scores,reason}]}`.
**System:** `GET /health`, `GET /meta` (counts, schema/model versions),
`GET /benchmarks`, `POST /benchmarks/run{suite}`.

## 9. Frontend data flow

- **Grid** — TanStack Virtual windowed grid over a flat `items` array from
  `useInfiniteQuery` keyed by `(filters, sort)`. Pages come from
  `GET /photos?cursor=…` using **keyset cursors** `(sort_at,id)` — where
  `sort_at = COALESCE(taken_at, mtime)` (not offset) — for stability under inserts. Total from `GET /photos/count` sizes the scrollbar; the
  virtualizer's visible range drives next-page fetch (overscan ~600 px).
- **Windowed client store** — only materialized pages are kept; far pages are LRU-
  evicted to bound memory at 100k. A date-bucket index endpoint backs the scrubber
  for jump-scroll.
- **Search/filter state** — one URL-synced store (query text, date range, people,
  tags, folders, `has_text`, sort). Any change swaps the query key → a fresh cursor
  stream from `/search` or `/photos` (text debounced 250 ms). Thumbnails via
  `<img src=/thumb/{id} loading=lazy>` (HTTP-cached). Selection state is separate.
- The Tauri shell hands React the sidecar base URL (localhost port) via a startup
  handshake.

## 10. Open risks & fallbacks

1. **HEIC decode perf/stability** (libheif slow, can segfault; 500 GB one-time).
   *Fallback:* process-pool isolation + per-file timeout+skip; alt decode path via
   Apple ImageIO/`sips`/CoreImage (faster on Apple Silicon); thumbs cached so decode
   is paid once.
2. **Face clustering quality at scale** (Chinese Whispers threshold sensitivity,
   over/under-merge on 300k+ faces). *Fallback:* HNSW kNN (avoid O(n²)), tunable
   thresholds, incremental pending pool, manual merge/split UI, block graph by
   time/quality.
3. **Tauri + PyInstaller packaging** (bundling ORT/CoreML/paddle/insightface/opencv;
   signing + notarization; sidecar lifecycle). *Fallback:* ship sidecar as a separate
   signed binary; PyInstaller `--onedir`; lazy model download on first run; validate
   notarization in Phase 0/1, not Phase 6; documented dev fallback of running the
   sidecar via `uv`.
4. **CoreML EP coverage/throughput** (op fallbacks to CPU, ANE quirks, uncertain
   batching gains). *Fallback:* benchmark EP-vs-CPU early, config to select EP, pin
   known-good opsets, keep CPU EP fallback.
5. **Filtered-HNSW latency/recall + SQLite single-writer during ingest.**
   *Fallback:* the three-tier strategy itself + empirically tuned thresholds;
   precomputed filter bitmaps; batched commits + `busy_timeout` for the writer;
   evaluate `usearch`/`faiss` if hnswlib filtering underperforms.
