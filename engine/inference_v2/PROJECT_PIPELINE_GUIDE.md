# Project Pipeline — Architecture, Naming & Step 12

> **Note:** written during initial development against the full training monorepo — `training-engine/` paths mean this `engine/` folder, and `training_pipeline/` paths referenced below no longer exist in this trimmed, public, inference-only repo (the few parity helpers that were needed now live in `engine/parity/`). Kept for architecture/design rationale.

Companion to [`README.md`](./README.md).

---

## 1. Training phases vs inference steps

Training phase order is **1 → 2 → 3** (artifact steps **11 → 12 → 13**):

```text
Phase 1 — token segmentation (B-SEG / I-SEG)     ← step 11  [locked]
Phase 2 — section divider   (B-PROJ_START)       ← step 12  [locked]
Phase 3 — segment classification                 ← step 13  [LOCKED]
```

At **inference runtime**, phase **2** runs before phase **1** when both are implemented (entry boundaries must exist before per-entry segmentation).

| Training phase | Training folder | Task | Inference step | Artifact | Inference module | Status |
|:--------------:|-----------------|------|---------------:|----------|------------------|--------|
| **1** | `project/phase1_token_segmentation` | Token segmentation | **11** | `11_project_segments.json` | `project_phase1_segment/` | 🔒 LOCKED |
| **2** | `project/phase2_section_divider` | Section divider | **12** | `12_project_boundaries.json` | `project_phase2_divider/` | 🔒 LOCKED |
| **3** | `project/phase3_segment_classification` | Segment classification | **13** | `13_project_fields.json` | `project_phase3_classify/` | 🔒 LOCKED |

### Naming rules

| Correct | Wrong / deprecated | Why |
|---------|-------------------|-----|
| `project_phase2_divider/` | `project_p1/` | Training **phase 1** is token segmentation, not boundaries |
| `run_project_phase2_divider()` | `run_project_phase1()` | Function name must match training phase 2 |
| `project_phase1_segment/` | putting segmentation in `project_p2/` | `phase1` in the folder name = training phase 1 |
| `12_project_boundaries.json` | `11_project_boundaries.json` | Boundaries are training phase **2** → artifact step **12** |

### Combined ONNX head names (separate system)

The combined ten-head bundle uses different indices (`project_phase1_logits` = boundaries). See `training_pipeline/COMBINED_MODEL_CONTRACT.md`. Standalone inference modules follow **training folder phase numbers**, not combined head indices.

---

## 2. Code isolation rules

1. **One stage per folder** — boundary logic only in `project_phase2_divider/`; segmentation in `project_phase1_segment/`; classification in `project_phase3_classify/`.
2. **No cross-section imports** — do not import `experience_phase*` from project modules or vice versa.
3. **No step 9 gap heuristics in boundaries** — never call `gap_heuristic.apply_entity_and_date_heuristics` from boundary stages (label id collision broke experience step 8).
4. **Shared helpers stay abstract** — `overlay_mongo_labels.py` loads `projectEntryHeads`; filtering `section in ("PROJECT", "PROJECTS")` stays in the project module.

---

## 3. Step 12 — `project_phase2_divider/` (training phase 2)

| Item | Value |
|------|-------|
| Entry point | `run_project_phase2_divider(tokens, resume_id)` |
| Checkpoint | `project_phase2_divider/best_model.pt` (from `training_pipeline/project/phase2_section_divider/saved_models/minilm/best_model.pt`) |
| Labels | `O`, `B-PROJ_START`, `I-PROJ_START` |
| GT mapper (UI) | `app/app/inference-v2/lib/step12BoundaryGt.ts` |
| Entry-line eval | `entryDividerLines` in artifact — same FBA table as training `per_resume_sparse/*.md` |
| Rerun | `POST /runs/{slug}/rerun/project_phase2_divider` |

**UI:** entry line comparison table (FBA) — like step 11 showed before the segmentation fix. Per-token `B-PROJ_START`/`I-PROJ_START` in `tokens[]` is secondary.

### Post-process pipeline

1. Model segment predictions on layout segments
2. Line-level collapse + `boundary_postprocess.py` (suppress/promote)
3. `style_heuristic.py` — dominant font signature alignment
4. `span_expand.py` — `I-PROJ_START` fill between heads

### Verification

Compare to `training_pipeline/project/phase2_section_divider/reports/minilm/per_resume_sparse/*.md` (Project Boundary Diagnostic).

---

## 4. Step 11 — `project_phase1_segment/` (training phase 1)

| Item | Value |
|------|-------|
| Entry point | `run_project_phase1_segment(tokens, resume_id)` |
| Checkpoint | `project_phase1_segment/best_model.pt` (from `training_pipeline/project/phase1_token_segmentation/saved_models/minilm/best_model.pt`) |
| Labels | `O`, `B-SEG`, `I-SEG` |
| Postprocess | Sequence continuity guard only (no `gap_heuristic`) |
| Rerun | `POST /runs/{slug}/rerun/project_phase1_segment` |

Consumes step **12** `B-PROJ_START` boundaries. Writes `segLabel` / `prediction` (`B-SEG`/`I-SEG`) on tokens; does not overwrite boundary `bioLabel`.

**UI:** per-token segmentation table (GT SEG vs pred) — matches `phase1_token_segmentation/reports/minilm/per_resume/*.md`. Entry-line FBA is step **12**, not step 11.

### Verification

Compare to `training_pipeline/project/phase1_token_segmentation/reports/minilm/per_resume/*.md`.

---

## 5. Step 13 — `project_phase3_classify/` (training phase 3)

| Item | Value |
|------|-------|
| Entry point | `run_project_phase3_classify(tokens, resume_id)` |
| Checkpoint | `project_phase3_classify/best_model.pt` (from `training_pipeline/project/phase3_segment_classification/saved_models/minilm/best_model.pt`) |
| Model | `PhraseSegmentClassifierModel` (MiniLM + 16D spatial + BiGRU) |
| Labels | `PROJECT_NAME`, `SDATE`, `EDATE`, `DESC` (DATE resolved post-inference) |
| Segment build | `construct_sentences_by_appearance` + `split_hyphenated_segments` |
| Postprocess | `postprocess_segment_predictions` + `resolve_dates_to_sdate_edate` |
| Rerun | `POST /runs/{slug}/rerun/project_phase3_classify` |
| UI | `13_project_fields.json` → `FieldClassificationView` (segment table) |
| Training parity | `training_pipeline/project/phase3_segment_classification/reports/minilm/per_resume/*.md` |

Consumes full token stream (steps 11–12 labels already on tokens). Stamps `bioLabel` with `B-PROJ_NAME`, `B-DESC`, `B-SDATE`, `B-EDATE`.

Verified on Karan: **100% segment accuracy** (10/10 segments).

---

## 6. Related files

```
training-engine/inference_v2/
├── PROJECT_PIPELINE_GUIDE.md      ← this file
├── project_phase1_segment/      ← step 11 (training phase 1) 🔒
├── project_phase2_divider/      ← step 12 (training phase 2) 🔒
└── project_phase3_classify/     ← step 13 🔒

training_pipeline/project/
├── phase1_token_segmentation/   ← training phase 1
├── phase2_section_divider/      ← training phase 2
└── phase3_segment_classification/ ← training phase 3
```
