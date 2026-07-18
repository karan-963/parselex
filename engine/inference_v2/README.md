# Inference V2 — Pipeline, Artifacts & Ground Truth

This folder runs the **end-to-end PDF inference pipeline** and writes numbered JSON artifacts under `training-engine/inference_runs/<slug>/`. The Next.js **inference-v2 visualizer** (`/app/inference-v2`) loads those artifacts and compares predictions to **live MongoDB labels**.

Read this before debugging “GT looks wrong” or “my viewer edits don’t show up”.

---

## Pipeline steps (master list)

**Training phase order** for Experience, Education, and Project is always:

```text
Phase 1 — token segmentation (B-SEG / I-SEG)
Phase 2 — section divider   (entry boundaries)
Phase 3 — segment classification (field labels)
```

**Personal** uses **atomic segment classification** (heuristic segmentation + segment classifier). **Skills** is direct token-level classification at step **7**.

**Inference artifact step numbers** follow training phase order for Project (11 → 12 → 13). Experience still uses phase 2 at step **8** and phase 1 at step **9** (historical artifact numbers).

Canonical catalog: `config.PIPELINE_STEPS` in `config.py`.

### Global & section (steps 1–3)

| Inf step | Artifact | Task | Training phase | Inference module | Status |
|---------:|----------|------|:--------------:|------------------|--------|
| **1** | `1_extracted_tokens.json` | PDF token extraction | — | `extract.py` | 🔒 LOCKED |
| **2** | `2_section_headings.json` | Heading detection | 1 | `section_p1/` | 🔒 LOCKED |
| **3** | `3_section_labels.json` | Section assignment | 2 | `section_p2/` | 🔒 LOCKED |

### Education (steps 4–6) — training phase 1 → 2 → 3

| Inf step | Artifact | Task | Training phase | Training folder | Inference module | Status |
|---------:|----------|------|:--------------:|-----------------|------------------|--------|
| **4** | `4_education_segments.json` | Token segmentation | **1** | `education/new_phase1_token_segmentation` | `education_phase1_segment/` | 🔒 LOCKED |
| **5** | `5_education_boundaries.json` | Section divider | **2** | `education/new_phase2_section_divider` | `education_phase2_divider/` | 🔒 LOCKED |
| **6** | `6_education_fields.json` | Segment classification | **3** | `education/new_phase3_segment_classification` | `education_phase3_classify/` | 🔒 LOCKED |

### Skills (step 7) — direct classification

| Inf step | Artifact | Task | Training folder | Inference module | Status |
|---------:|----------|------|-----------------|------------------|--------|
| **7** | `7_skills_fields.json` | Skills token BIO | `skills/` | `skills_classify/` | 🔒 LOCKED |

### Experience (steps 8–10) — training phase 1 → 2 → 3

Listed in **training phase order**. At inference runtime, phase **2** runs before phase **1** (boundaries must exist before per-entry segmentation).

| Inf step | Artifact | Task | Training phase | Training folder | Inference module | Status |
|---------:|----------|------|:--------------:|-----------------|------------------|--------|
| **8** | `8_experience_segments.json` | Token segmentation | **1** | `experience/phase1_token_segmentation` | `experience_phase1_segment/` | 🔒 LOCKED |
| **9** | `9_experience_boundaries.json` | Section divider | **2** | `experience/phase2_section_divider` | `experience_phase2_divider/` | 🔒 LOCKED |
| **10** | `10_experience_classification.json` | Segment classification | **3** | `experience/phase3_segment_classification` | `experience_phase3_classify/` | 🔒 LOCKED |

### Project (steps 11–13) — training phase 1 → 2 → 3

| Inf step | Artifact | Task | Training phase | Training folder | Inference module | Status |
|---------:|----------|------|:--------------:|-----------------|------------------|--------|
| **11** | `11_project_segments.json` | Token segmentation | **1** | `project/phase1_token_segmentation` | `project_phase1_segment/` | 🔒 LOCKED |
| **12** | `12_project_boundaries.json` | Section divider | **2** | `project/phase2_section_divider` | `project_phase2_divider/` | 🔒 LOCKED |
| **13** | `13_project_fields.json` | Segment classification | **3** | `project/phase3_segment_classification` | `project_phase3_classify/` | 🔒 LOCKED |

### Personal (step 15) — segment classification

| Inf step | Artifact | Task | Training folder | Inference module | Status |
|---------:|----------|------|-----------------|------------------|--------|
| **15** | `15_personal_fields.json` | Personal entity BIO | `personal/` | `personal_classify/` | 🔒 LOCKED |

### Finalize

| Inf step | Artifact | Task | Inference module | Status |
|---------:|----------|------|------------------|--------|
| **14** | `14_final_classified_tokens.json`, `structured.json` | Final merge | `entities.py` | 🔒 LOCKED |

**Guides:** [`EXPERIENCE_PIPELINE_GUIDE.md`](./EXPERIENCE_PIPELINE_GUIDE.md) · [`PROJECT_PIPELINE_GUIDE.md`](./PROJECT_PIPELINE_GUIDE.md)

Do **not** change locked inference logic, GT derivation, or UI for locked steps without explicit re-open.

---

## Inference execution order (orchestrator)

`pipeline.py` runs **implemented** stages only:

```text
1  extract
2  section_p1 (headings)
3  section_p2 (section labels)
5  education_phase2_divider     ← training phase 2
4  education_phase1_segment      ← training phase 1
6  education_phase3_classify      ← training phase 3
7  skills_classify                ← direct classification
8  experience_phase2_divider   ← training phase 2
9  experience_phase1_segment    ← training phase 1
10 experience_phase3_classify   ← training phase 3
12 project_phase2_divider       ← training phase 2
11 project_phase1_segment       ← training phase 1
13 project_phase3_classify      ← training phase 3
15 personal_classify            ← atomic segment classification
14 finalize
```

Personal step **15** runs after project phase 3 and before finalize. Segmentation uses `build_personal_segments` (not `construct_sentences_by_appearance`). GT uses `derive_segment_label` with `segment_labels_match` (B vs I equivalence).

---

## Terminology

| Concept | Meaning |
|---------|---------|
| **Training phase 1** | Token segmentation — combine tokens into phrase segments (`B-SEG` / `I-SEG`) |
| **Training phase 2** | Section divider — group segments into entries (`B-ENTRY`, `B-PROJ_START`, `B-EDU_START`, …) |
| **Training phase 3** | Segment classification — label each phrase block (`ROLE`, `COMP`, `DEGREE`, …) |
| **Inference step** | Artifact file number in `inference_runs/<slug>/` |
| **Direct classification** | Skills — single token-level model pass |
| **Segment classification** | Personal — heuristic atomic segments + segment classifier |

### Experience labels by phase

| Training phase | Inf step | Labels |
|:--------------:|---------:|--------|
| 1 | 8 | `B-SEG` / `I-SEG` |
| 2 | 9 | `B-ENTRY` / `I-ENTRY` |
| 3 | 10 | `ROLE` / `COMP` / `DATE` / `DESC` |

Step 9 UI entry-line eval uses `experienceEntryHeads` (GT) vs step 9 `B-ENTRY` lines (pred). Step 8 `tokens[]` hold phrase-seg labels for step 10.

### Project labels by phase

| Training phase | Inf step | Labels |
|:--------------:|---------:|--------|
| 1 | 11 | `B-SEG` / `I-SEG` |
| 2 | 12 | `B-PROJ_START` / `I-PROJ_START` |
| 3 | 13 | `PROJECT_NAME` / `SDATE` / `EDATE` / `DESC` |

### Skills labels (step 7)

| Inf step | Labels |
|---------:|--------|
| 7 | `O`, `B-SKILL`, `I-SKILL`, `B-SKILL_TYPE`, `I-SKILL_TYPE` |

Direct token-level classification — no segmentation/divider chain. GT: MongoDB `bioLabel` mapped to 5-class BIO. Eval: alphanumeric tokens only (matches `skills/reports/minilm/`).

### Education labels by phase (pending)

| Training phase | Inf step | Labels |
|:--------------:|---------:|--------|
| 1 | 4 | `B-SEG` / `I-SEG` |
| 2 | 5 | `B-EDU_START` / `I-EDU_START` |
| 3 | 6 | `DEGREE` / `INSTITUTION` / `DATE` / … |

**Reports:** step 9/12/5 → `phase2_section_divider/reports/` · step 8/11/4 → `phase1_token_segmentation/reports/` · step 10/13/6 → `phase3_segment_classification/reports/`

---

## Quick reference: where GT comes from

| Artifact | Step | MongoDB source | GT column in UI | Notes |
|----------|------|----------------|-----------------|-------|
| `9_experience_boundaries.json` | Experience divider | `tokens[].bioLabel` | `step8GroundTruthLabel()` | **Locked:** `app/app/inference-v2/lib/step8BoundaryGt.ts` |
| `8_experience_segments.json` | Experience segmentation | `tokens[].bioLabel` | `getGtSegmentLabel` → `B-SEG` / `I-SEG` | UI: **phrase group table** (not per-token) |
| `6_*_fields.json` | Education classification | `tokens[].bioLabel` | Raw `bioLabel` | Block-level FBA in `blockClassification` |
| `7_*_fields.json` | Skills classification | `tokens[].bioLabel` | Raw `bioLabel` (5-class) | Alphanumeric eval in `tokenClassification` |
| `10_experience_classification.json` | Experience classification | `tokens[].bioLabel` | Raw `bioLabel` | `B-COMP`, `I-ROLE`, `B-SDATE`, … |
| `12_*_boundaries.json` | Project divider | `tokens[].bioLabel` | `step12GroundTruthLabel()` | **Locked:** `app/app/inference-v2/lib/step12BoundaryGt.ts` |
| `2_section_headings.json` | Section headings | `tokens[].bioLabel` | Heading text match | `B-HEADING` / `I-HEADING` |
| `3_section_labels.json` | Section assignment | `tokens[].section` | Per-chunk section | Uses pipeline heading map |

**MongoDB fetch:** `GET /api/resumes/{resumeId}` — always fresh (`cache: 'no-store'`). Use **Refresh GT** in the artifact panel after saving labels in `/viewer/{resumeId}`.

**Token alignment key (inference ↔ MongoDB):**

```
(page, round(x0, 2), round(y0, 2))
```

Never match by `lineIndex` or `tokenIndex` alone — extraction and MongoDB use different index schemes.

---

## Rerun a single stage

```bash
# From repo root, with training-engine on PYTHONPATH
curl -X POST http://localhost:8000/inference-v2/runs/Karan_f71781/rerun/education_phase2_divider
curl -X POST http://localhost:8000/inference-v2/runs/Karan_f71781/rerun/education_phase1_segment
curl -X POST http://localhost:8000/inference-v2/runs/Karan_f71781/rerun/education_phase3_classify
curl -X POST http://localhost:8000/inference-v2/runs/Karan_f71781/rerun/skills_classify
curl -X POST http://localhost:8000/inference-v2/runs/Karan_f71781/rerun/experience_phase2_divider
curl -X POST http://localhost:8000/inference-v2/runs/Karan_f71781/rerun/experience_phase1_segment
curl -X POST http://localhost:8000/inference-v2/runs/Karan_f71781/rerun/experience_phase3_classify
curl -X POST http://localhost:8000/inference-v2/runs/Karan_f71781/rerun/project_phase2_divider
curl -X POST http://localhost:8000/inference-v2/runs/Karan_f71781/rerun/project_phase1_segment
curl -X POST http://localhost:8000/inference-v2/runs/Karan_f71781/rerun/project_phase3_classify
curl -X POST http://localhost:8000/inference-v2/runs/Karan_f71781/rerun/personal_classify
```

Deprecated aliases still work: `experience_p1` → `experience_phase2_divider`, `experience_p2` → `experience_phase1_segment`, `experience_p3` → `experience_phase3_classify`, `project_p1` → `project_phase2_divider`, `project_p3` → `project_phase3_classify`.

Python changes only affect predictions after rerun. **GT always comes from MongoDB** — no rerun needed for label edits.

---

## Step 8 ground truth (LOCKED)

**Single source:** `app/app/inference-v2/lib/step8BoundaryGt.ts`

| MongoDB `bioLabel` | Step 8 GT |
|--------------------|-----------|
| `O`, `B-HEADING`, `I-HEADING` | `O` |
| `B-COMP`, `B-ROLE`, `B-SDATE`, `B-EDATE`, `B-DESC`, … | `B-ENTRY` |
| `I-COMP`, `I-ROLE`, `I-DESC`, … | `I-ENTRY` |
| `B-ENTRY` | `B-ENTRY` |

Python mirror: `scratch/export_mongo_boundary_gt.py` → `bio_to_boundary_label()` (keep in sync).

**Do not** use `experienceEntryHeads` for step 9 token GT — that strategy is for step 9 entry-line FBA only.

---

## Step 8–10 ground truth

### Entry divider eval (`9_experience_boundaries.json`)

**Ground truth:** MongoDB `experienceEntryHeads` — entry start lines (training `phase2_section_divider`).

**Predictions:** Step 9 `B-ENTRY` lines from `9_experience_boundaries.json`.

**JSON `tokens[]`:** Per-token `B-ENTRY`/`I-ENTRY` boundary labels.

### Token segmentation (`8_experience_segments.json`)

**JSON `tokens[]`:** `B-SEG`/`I-SEG` from `experience_phase1_segment`, consumed by step 10.

### Segment classification (`10_experience_classification.json`)

**Task:** Classify phrase blocks as `ROLE`, `COMP`, `DATE`, or `DESC`.

**Inference chain:**
1. Group tokens by step 9 `B-ENTRY` divider lines.
2. Split each entry into blocks at step 8 `B-SEG` dividers plus line/delimiter heuristics.
3. Classify each block with `ResumeChunkClassifier`.
4. Write per-token field BIO to `tokens[].prediction`.

**Rerun:** Requires steps 8 and 9 artifacts replayed (`rerun/experience_phase3_classify`).

---

## Module layout

```
inference_v2/
├── pipeline.py                  # Orchestrator (implemented stages only)
├── routes.py                    # FastAPI rerun endpoints
├── config.py                    # PIPELINE_STEPS catalog + STAGE_ARTIFACTS
├── storage.py
├── extract.py                   # Step 1
├── section_p1/                  # Step 2 — section phase 1
├── section_p2/                  # Step 3 — section phase 2
├── education_phase1_segment/    # Step 4 — education phase 1 🔒
├── education_phase2_divider/    # Step 5 — education phase 2 🔒
├── education_phase3_classify/   # Step 6 — education phase 3 🔒
├── skills_classify/             # Step 7 — skills direct classify 🔒
├── experience_phase1_segment/   # Step 8 — experience phase 1 🔒
├── experience_phase2_divider/   # Step 9 — experience phase 2 🔒
├── experience_phase3_classify/  # Step 10 — experience phase 3 🔒
├── project_phase1_segment/      # Step 11 — project phase 1 🔒
├── project_phase2_divider/      # Step 12 — project phase 2 🔒
├── project_phase3_classify/     # Step 13 — project phase 3 🔒
├── personal_classify/           # Step 15 — personal segment classify 🔒
```

---

## Common pitfalls

| Symptom | Likely cause | See |
|---------|----------------|-----|
| GT still shows old labels | Did not click **Refresh GT** or save in viewer | README § verify GT |
| Step 8 GT mostly `O` | Wrong strategy (`experienceEntryHeads`) | README § Step 8 GT |
| `Present` / `July'22` as `B-ENTRY` | Step 9 heuristics applied in step 8 | [Guide §4–5.2](./EXPERIENCE_PIPELINE_GUIDE.md) |
| Too many jobs | Entry grouping uses every `B-ENTRY` line | [Guide §2, §5.3](./EXPERIENCE_PIPELINE_GUIDE.md) |
| Block text merges date + project line | `split_entry_blocks` line boundaries | [Guide §5.6](./EXPERIENCE_PIPELINE_GUIDE.md) |
| Predictions stale after Python fix | Rerun the stage — artifacts are not auto-regenerated | README § Rerun |

---

## Related docs

- **Experience steps 8–10:** [`EXPERIENCE_PIPELINE_GUIDE.md`](./EXPERIENCE_PIPELINE_GUIDE.md)
- **Project steps 11–13:** [`PROJECT_PIPELINE_GUIDE.md`](./PROJECT_PIPELINE_GUIDE.md)
- Training pipelines: `training_pipeline/experience/EXPERIENCE_PIPELINE.md`, `education/EDUCATION_PIPELINE.md`, `project/PROJECT_PIPELINE.md`
- Dataset alignment: root `CLAUDE.md` § Dataset Alignment
- Labeling schema: `app/docs/LABELING_PLAN.md`
