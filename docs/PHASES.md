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

> **Status.** Detector = insightface **buffalo_l** (SCRFD det_10g + ArcFace w600k_r50,
> 512-d), downloaded on first use; `uv sync --extra faces`. Unlike CLIP text, SCRFD+
> ArcFace run fine on the CoreML EP. Faces are detected on the **re-decoded original**
> (downscaled to `face_max_edge`), and the faces stage runs **in the ingest thread**
> (a §2 simplification vs. a separate faces process). insightface/cv2 are imported
> lazily, so CI stays on `--extra embed`. People UI in-app check is a user-run like the grid.

- [x] SCRFD detect + align + ArcFace embed → `faces` + `faces.f32` memmap. — `faces/detector.py`, `ingest/orchestrator.py:_run_faces`
      *Done:* faces with normalized bbox/landmarks/quality/embed_row written (test_faces_stage).
- [x] Faces HNSW + Chinese Whispers clustering → `clusters`; reps chosen. — `faces/cluster.py`
      *Done:* clustering recovers synthetic groups; reps = highest-quality face (tests).
- [x] Incremental assign + pending pool + `/people` endpoints + rename/merge/split UI. — `faces/service.py`, `api/people.py`, `frontend/src/components/PeoplePanel.tsx`
      *Done:* new faces join existing people by centroid (no full re-cluster); merge/split/rename API tested; People UI builds.
- [x] **Benchmark:** face det+embed throughput + cluster purity → `benchmarks`. — `scripts/bench_faces.py`
      *Done (real run):* `face_throughput≈76 faces/s` (CoreML). `cluster_purity=1.0` was measured under **photometric** variation only (lighting/blur, same pose) — it did **not** cover pose, which is the real failure mode. Pose robustness is addressed by recalibrating the edge/assign thresholds (0.50/0.60 → 0.35/0.42) from the measured stranger-cosine ceiling (≤0.21); see `test_edge_threshold_controls_pose_merging`. A labeled multi-pose purity benchmark is a Phase 6 item.

## Phase 4 — Grouping + OCR/FTS + tags

> **Status.** All four groupers, OCR/FTS, tags, and search fusion are built and green
> (65 backend tests). **Two documented divergences from the plan** (both keep the app
> local-first + CI lean; see ARCHITECTURE §5/§6): OCR uses **Apple Vision** (`ocrmac`),
> not PaddleOCR (paddlepaddle is ARM-Mac-flaky); semantic themes use the **Chinese
> Whispers kNN clusterer** already shipped for faces, not HDBSCAN (avoids a scikit-learn
> dependency). OCR is macOS-only and, like faces, **skips gracefully on CI/Linux**
> (`uv sync --extra embed`), so both stages stay out of CI. Grouping runs as a batch
> `_run_grouping` pass at the end of ingest. Group reps used a resolution proxy at Phase
> 4 and now use the Phase 5 `quality`/`aesthetic` scores (scoring runs before grouping),
> falling back to resolution when scores are absent (`grouping.service._rank_key`).

- [x] Near-dup (phash+cosine), burst (time+camera), event (time/GPS) groupers → `groups`/`group_items`. — `grouping/{near_dup,events,service}.py`, `db/groups.py`
      *Done:* `GroupingService.rebuild` writes all layers; near-dup pairs the exact-dup set, bursts collapse to a resolution-best rep with ranked members (test_grouping).
- [x] Semantic clustering (Chinese Whispers over CLIP kNN; HDBSCAN deviation, §5) → `semantic` groups. — `grouping/semantic.py`, `api/groups.py`, `frontend/src/components/GroupsPanel.tsx`
      *Done:* themes form and are browsable via `GET /groups?kind=semantic` and the Groups view.
- [x] Apple Vision OCR → `ocr` (FTS5) + `ocr_regions`; FTS fused into `/search` (RRF) + tags endpoints. — `ocr/{engine,service}.py`, `db/{ocr,tags}.py`, `api/{search,tags}.py`
      *Done:* text visible in a photo is searchable (`test_ocr`, `test_search_fusion`); `has_text` filter + tags CRUD tested (`test_api_tags`).
- [x] **Benchmark/test:** grouping precision/recall + OCR recall → `benchmarks`. — `scripts/bench_grouping.py`, `scripts/bench_ocr.py`
      *Done (real runs):* near-dup `grouping_f1=1.0` + `event_recall=1.0` at `≈1976 photos/s` on a **clean synthetic** near-dup set (12 scenes × 4 JPEG variants — no adversarial collisions; a labeled real-photo set is a Phase 6 item). Apple Vision `ocr_recall=0.966` (28/29 words) at `≈7.9 img/s` on rendered-text images (throughput includes first-call model warmup; small bitmap font).
- [x] **Date & location search** (follow-up, ARCHITECTURE §4). — `query_parse.py`, `geo/`, `search.py`, `api/search.py`, `frontend/src/components/SearchBar.tsx`
      *Done:* NL date parsing ("beach 2024", "last summer") + explicit date range; offline place-name geocoding (geonamescache, `geo` extra) + GPS radius / "near this photo"; filter-only browse when the query is just a date/place; UI filters row + applied-filter chips. Tested by `test_query_parse`, `test_geo`, `test_search_location` (all CI-safe via fakes). Not yet: camera/person/tag filters, region-qualified place disambiguation.

## Phase 5 — Trip triage & aesthetics
- [x] Aesthetic + quality scorers populate `aesthetic`/`quality` (ingest `score` stage,
      `scored_at` marker, migration 0003).
      *Done:* deterministic numpy/Pillow scorers (`iris/scoring/quality.py`) — **not** the
      LAION MLP; see the ARCHITECTURE §7 build note (LAION needs ViT-L/14 768-d embeddings,
      Iris embeds ViT-B/32 512-d). No cv2/no model, so it runs in embed-only CI. The
      orchestrator test asserts both columns + `scored_at` set for the decodable photos.
- [x] Four-score computation + percentile-rank normalization + MMR + preset profiles + `/triage`.
      *Done:* `POST /triage{group_id|date_range, preset}` returns a near-dup/burst-collapsed,
      MMR-diversified shortlist with per-score breakdown + a human `reason`; three presets
      (`story-ready`, `print-worthy`, `delete-candidates`) via `GET /triage/presets`.
- [x] Triage UI (event view, preset switch, accept/reject).
      *Done:* `TriagePanel` — event picker, preset tabs (switching visibly re-ranks the same
      event), per-photo score bars + reason + dup flag, local keep/reject verdicts.
- [x] **Benchmark/test:** triage scoring throughput + overlap vs. a known gold pick → `benchmarks`.
      *Done (real run, `scripts/bench_triage.py`, phase="5"):* `triage_overlap=1.0`
      (15/15 sharp "hero" frames kept over their blurred near-dup variants — gold is which
      frame we objectively drew sharpest) at `triage_throughput≈46,000 photos/s` through the
      full four-score + normalize + collapse + MMR pipeline (CLIP vectors are stand-ins;
      triage only reads stored vectors). A labeled real-photo gold set is a Phase 6 item.

## Phase 6 — Packaging, scale-hardening & polish

> **Scope note.** This phase splits into *software hardening* (fully built + tested here)
> and *resource-gated delivery* — a signed/notarized build needs an Apple Developer cert
> and the Rust toolchain; the headline scale numbers need the real ~100k/500 GB library.
> The build **configuration** and the **e2e harness** are committed; the two boxes that
> require those resources are marked **[ ] (you run)** with the exact commands.

- [x] LRU preview-cache eviction + memmap compaction + crash-resume soak.
      *Done:* `GET /preview/{id}` renders 1024 px WebP on demand (read-only) and LRU-evicts
      to `preview_cache_gb`; `GET /maintenance/storage` reports thumb/preview/embeddings/db
      usage vs. caps; `POST /maintenance/compact` reclaims orphaned `clip.f32`/`faces.f32`
      rows + renumbers `embed_row` + rebuilds the index (409 while ingest runs). Soak test
      `test_soak.py` kill-storms ingest (repeated cancel/resume) then asserts every stage
      completes, no duplicate rows, and `PRAGMA integrity_check = ok`. (`test_previews`,
      `test_compaction`, `test_api_maintenance`, `test_soak`.) Also fixed an hnswlib
      segfault on saving an empty index (surfaced by compacting away every vector).
- [ ] **(you run)** PyInstaller sidecar (`--onedir`) bundling ORT/CoreML/opencv/insightface; lazy model download.
      *Scaffolded:* `backend/iris-sidecar.spec` (+ `packaging/entrypoint.py`) and
      `scripts/build_sidecar.sh`, which builds and **smoke-tests that the frozen binary
      prints `IRIS_SIDECAR_READY` with no dev Python**, then stages it into `src-tauri`.
      Run `scripts/build_sidecar.sh` on a Mac with the ML extras. *Done when:* it starts
      with no dev Python present.
- [ ] **(you run)** Tauri release bundle + code signing + notarization.
      *Scaffolded:* `tauri.conf.json` `bundle.macOS` (entitlements + `signingIdentity`) +
      `entitlements.plist` (hardened runtime: disable library validation for the bundled
      native dylibs) + `docs/PACKAGING.md` runbook. Needs Rust + an Apple Developer cert
      (neither present here). *Done when:* a signed, notarized `.dmg` launches on a clean Mac.
- [x] End-to-end benchmark harness (`scripts/bench_e2e.py`) — real ingest, `e2e_ingest` +
      `peak_rss` + `search_p95_full`, logged to `benchmarks` (phase 6). Smoke-verified on a
      synthetic set (records real, small-scale rows).
- [ ] **(you run)** Full-scale ~100k / 500 GB run → headline numbers.
      *Done when:* `bench_e2e.py --root <library> --embed` records `e2e_ingest`, `peak_rss`,
      `search_p95_full` from the real run (never fabricated — CLAUDE.md #2). Then profile +
      tune tier thresholds / batch sizes and commit them.
