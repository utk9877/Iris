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

> **Status.** Backend complete & verified (22 tests, ruff/mypy --strict green). Schema
> 0002 added `roots` + `sort_at` (keyset key; `taken_at` is nullable) — see ARCHITECTURE
> §1/§9. Hash+EXIF are fused into one thread-pool task (both read the file once). The
> React grid is built & typechecks (`npm run build`) but its in-app scroll is
> **runtime-unverified** pending a real `tauri dev` run with a photo folder.

- [x] Scan walker → `photos` rows; re-scan is idempotent. — `ingest/scanner.py`, `db/photos.py:upsert_scanned`
      *Done:* `upsert_scanned` ON CONFLICT(path) resets stage markers only when size/mtime change (test_orchestrator).
- [x] Hash (blake3) + EXIF (thread pool) populate identity + metadata. — `ingest/metadata.py`
      *Done:* `content_hash`/dims/EXIF set; `hashed_at`/`exif_at` advance (unit + e2e tests).
- [x] Decode+thumb spawn process pool + HEIC; content-addressed webp; phash piggybacked. — `ingest/thumbnails.py`
      *Done:* sharded webp written atomically; **HEIC verified**; bad files skipped (ok=False), not fatal.
- [x] Single-writer + job manager + crash-resume (re-enqueue by `*_at IS NULL`). — `ingest/orchestrator.py`
      *Done:* `test_resume_only_processes_pending` proves a restart re-processes only the pending photo.
- [x] `GET /photos` (keyset) + `/thumb` + `/photos/count`; React virtual grid + native folder picker. — `api/photos.py`, `frontend/src/components/PhotoGrid.tsx`, `ScanBar.tsx`
      *Done & runtime-verified in-app:* Choose-folder → scan → thumbnails render in the TanStack-Virtual grid. Grid is virtualized (row-windowed); true 5k smooth-scroll is a scale check to repeat on a large library.
- [x] **Benchmark:** ingest throughput + decode p50 → `benchmarks`. — `scripts/bench_ingest.py`
      *Done (synthetic dataset):* `ingest_throughput` and `decode_p50` recorded from a real run (git_sha-tagged). Real-library numbers are a Phase 6 task.

## Phase 2 — Embeddings & semantic search

> **Status.** Model = **CLIP ViT-B/32 (512-d)**, pre-exported ONNX from HuggingFace,
> downloaded on first use (config `embed_model`). **CoreML finding (§10 #4):** the
> vision encoder runs on CoreML, but the **text encoder hard-fails on CoreML at
> inference** (value-dependent) → text runs on CPU (tiny, once per query). Deps split
> into `embed`/`faces`/`ocr` extras; Phase 2 installs `uv sync --extra embed`. Embedding
> is done from the stored thumbnail (a simplification vs. §2's shared-memory hand-off).
> Grid search UI compiles; in-app "dog on beach" relevance is a user-run check.

- [x] CLIP ONNX embed stage (CoreML vision), batched; `clip.f32` memmap + `embed_row`. — `embeddings/clip.py`, `ingest/orchestrator.py:_run_embed`
      *Done:* embed stage sets `embed_at` + a memmap row for every thumbnailed photo (test_embed_stage).
- [x] Build/persist hnswlib index; rebuildable from memmap. — `embeddings/index.py`, `embeddings/service.py`
      *Done:* deleting `clip.hnsw` → a fresh service rebuilds from memmap + DB (tested).
- [x] Text-query embedding + three-tier search (§4) + RRF hook (FTS = Phase 4). — `search.py`, `api/search.py`
      *Done:* `POST /search` returns ranked ids + `tier`; all 4 tiers unit-tested.
- [x] Search UI: query box wired to `/search`, results render in the grid. — `frontend/src/components/SearchBar.tsx`, `PhotoGrid.tsx`
      *Done (builds/typechecks);* in-app relevance is a user-run check like the grid.
- [x] **Benchmark:** embed throughput + search p50/p95 + recall@10 vs. brute-force → `benchmarks`. — `scripts/bench_embed_search.py`
      *Done (synthetic):* real run logged `embed_throughput≈115 img/s` (CoreML), `search_p50≈7.7 ms`, `recall_at_10=1.0` (perfect at 800; expect <1 at 100k → Phase 6).

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
