# Delivery phases

Dependency-ordered, one capability per phase. Every task has a **done-when**; every
phase ends with a **benchmark/test** task that proves the capability (results logged
to `benchmarks`, never fabricated).

## Phase 0 — Foundations & harness

> **Status (in progress).** Deviation from the original wording, agreed this session:
> the heavy ML stack is deferred to an optional `ml` extra (Phase 2), so Phase 0
> installs only the lean core — this decouples the foundation/CI from the
> paddlepaddle/insightface Apple-Silicon build issues (ARCHITECTURE §10). The two
> environment tasks below (Rust toolchain, CoreML EP) are therefore deferred/partial.

- [~] ~~Install Rust toolchain; `uv sync --all-extras`~~; confirm ORT CoreML EP imports.
      *Deferred:* Rust install owned by the user (Tauri glue is written but unbuilt);
      lean `uv sync` done; CoreML EP verification moves to Phase 2 with the `ml` extra.
- [x] App data dir + config module (paths from §3, env overrides). — `src/iris/config.py`
      *Done:* running the sidecar creates the data-dir tree (verified via `IRIS_DATA_DIR`).
- [x] SQLite migration runner + **schema v1** (§1) with WAL + `busy_timeout`. — `src/iris/db/`
      *Done:* fresh DB migrates to v1; `PRAGMA journal_mode` returns `wal` (test_db.py). FTS5 confirmed present.
- [x] FastAPI skeleton + `GET /health`, `GET /meta`, `GET /` landing + CORS; Tauri launches sidecar via handshake. — `src/iris/app.py`, `frontend/src-tauri/src/lib.rs`, `frontend/src/App.tsx`
      *Done & runtime-verified:* Tauri app shows a green "Sidecar connected" panel with schema v1.
      Hardening: sidecar launched from the venv binary directly (no orphaned grandchild), **ephemeral port** (real port via handshake, no collisions), handshake emitted only after the socket binds, CORS for the dev + `tauri://` origins.
- [x] Benchmark harness: `benchmarks` writer + `POST /benchmarks/run` + CI. — `src/iris/benchmarks.py`, `.github/workflows/ci.yml`
      *Done:* smoke suite records a **real** measured DB round-trip; ruff/mypy --strict/pytest green locally.
- [x] **Test:** migration + health integration test; smoke bench row proves the harness. — `tests/`
      *Done:* 8 tests pass; a `smoke_db_roundtrip` row is recorded and read back end-to-end.

## Phase 1 — Ingest & storage (metadata + thumbs + grid)
- [ ] Scan walker → `photos` rows (path/dir/size/mtime); re-scan is idempotent.
      *Done when:* scanning a folder twice yields no duplicate rows.
- [ ] Hash (blake3) + EXIF stages (thread pool) populate identity + metadata.
      *Done when:* `content_hash`, `taken_at`, camera fields set; `*_at` markers advance.
- [ ] Decode+thumb process pool (spawn) with HEIC support; content-addressed webp writes; phash piggybacked.
      *Done when:* thumbs land at sharded paths; bad HEIC is skipped, not fatal.
- [ ] Single-writer + job manager + crash-resume (re-enqueue by `*_at IS NULL`).
      *Done when:* killing mid-ingest and restarting completes only unfinished photos.
- [ ] `GET /photos` (keyset cursor) + `/thumb` + `/photos/count`; React virtual grid renders thumbnails.
      *Done when:* a 5k-photo folder scrolls smoothly in the grid.
- [ ] **Benchmark:** ingest throughput (img/s) + thumb decode p50 on a real folder → `benchmarks`.
      *Done when:* rows `ingest_throughput`, `decode_p50` recorded from an actual run.

## Phase 2 — Embeddings & semantic search
- [ ] CLIP/SigLIP ONNX embed worker (CoreML EP), batched; write `clip.f32` memmap + `embed_row`.
      *Done when:* every non-missing photo has `embed_at` set and a memmap row.
- [ ] Build/persist hnswlib clip index; rebuildable from memmap on startup.
      *Done when:* deleting `clip.hnsw` and restarting rebuilds it from memmap.
- [ ] Text-query embedding + three-tier search (§4) + RRF fusion with FTS placeholder.
      *Done when:* `POST /search{query}` returns ranked ids with the chosen `tier`.
- [ ] Search UI: query box + filter state store wired to `/search`.
      *Done when:* typing "dog on beach" returns relevant photos in the grid.
- [ ] **Benchmark:** embed throughput (img/s) + search p50/p95 + recall@10 vs. brute-force on a labeled query set → `benchmarks`.
      *Done when:* `embed_throughput`, `search_p50`, `recall@10` recorded from a real run.

## Phase 3 — Faces & people
- [ ] SCRFD detect + align + ArcFace embed workers → `faces` + `faces.f32` memmap.
      *Done when:* faces with bbox/landmarks/quality/embed_row exist for a test set.
- [ ] Faces HNSW + Chinese Whispers clustering → `clusters`; reps chosen.
      *Done when:* clusters form and each has a rep face.
- [ ] Incremental assign + pending pool + `/people` endpoints + rename/merge/split UI.
      *Done when:* importing new photos assigns known people without full re-cluster.
- [ ] **Benchmark/test:** face det+embed throughput + cluster purity/recall on a hand-labeled sample → `benchmarks`.
      *Done when:* `face_throughput` and `cluster_purity` recorded from a real run.

## Phase 4 — Grouping + OCR/FTS + tags
- [ ] Near-dup (phash+cosine), burst (time+camera), event (time/GPS) groupers → `groups`/`group_items`.
      *Done when:* a burst folder collapses to one rep with ranked members.
- [ ] Semantic clustering (HDBSCAN over CLIP kNN) → `semantic` groups.
      *Done when:* themes appear and are browsable via `/groups?kind=semantic`.
- [ ] PaddleOCR worker → `ocr` (FTS5) + `ocr_regions`; wire FTS into search fusion + tags endpoints.
      *Done when:* searching text visible in a photo returns it; tags CRUD works.
- [ ] **Benchmark/test:** grouping precision/recall on a labeled set + OCR search recall → `benchmarks`.
      *Done when:* `grouping_f1` and `ocr_recall` recorded from a real run.

## Phase 5 — Trip triage & aesthetics
- [ ] LAION aesthetic + OpenCV quality scorers populate `aesthetic`/`quality`.
      *Done when:* both columns are set for a test event.
- [ ] Four-score computation + percentile-rank normalization + MMR + preset profiles + `/triage`.
      *Done when:* `POST /triage{event,preset}` returns a deduped, diversified shortlist with reasons.
- [ ] Triage UI (event view, preset switch, accept/reject).
      *Done when:* switching presets visibly re-ranks the same event.
- [ ] **Benchmark/test:** triage scoring throughput + shortlist stability + overlap vs. a manual gold pick → `benchmarks`.
      *Done when:* `triage_throughput` and `triage_overlap` recorded from a real run.

## Phase 6 — Packaging, scale-hardening & polish
- [ ] PyInstaller sidecar (`--onedir`) bundling ORT/CoreML/opencv/insightface/paddle; lazy model download.
      *Done when:* the packaged sidecar starts with no dev Python present.
- [ ] Tauri release bundle + code signing + notarization.
      *Done when:* a signed, notarized `.app`/`.dmg` launches on a clean Mac.
- [ ] LRU preview-cache eviction + memmap compaction + crash-resume soak.
      *Done when:* caches respect configured caps; a kill-storm leaves the DB consistent.
- [ ] Full-scale run on the real ~100k / 500 GB library; profile and tune tier thresholds + batch sizes.
      *Done when:* a complete ingest finishes and tuned thresholds are committed.
- [ ] **Benchmark:** end-to-end 100k ingest time, peak memory, search p95 at full scale → `benchmarks`.
      *Done when:* `e2e_ingest`, `peak_rss`, `search_p95_full` recorded from the real 100k run.
