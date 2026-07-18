# Experience Pipeline — Architecture, Isolation & Known Failure Points

**Read this before changing steps 8–10.** Training phases are **1 → 2 → 3** (segment → divide → classify). Inference **executes** phase 2 before phase 1 (step 8 before step 9) because segmentation runs per entry slice.

**Companion:** [`README.md`](./README.md) (artifacts, GT sources, rerun commands).

---

## 1. The three-stage chain (training ↔ inference)

Experience parsing is **three independent stages**. Each stage has its own training folder, inference module, artifact, and label vocabulary. **Never mix responsibilities across stages.**

```
PDF tokens (EXPERIENCE section)
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 8 — DIVIDE (entry boundaries)                               │
│  Training:  experience/phase2_section_divider                     │
│  Inference: experience_phase2_divider/                                        │
│  Artifact:  9_experience_boundaries.json                          │
│  Labels:    O | B-ENTRY | I-ENTRY                                 │
│  Question:  "Which lines belong to the same job entry?"           │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 9 — SEGMENT (phrase blocks within each entry)               │
│  Training:  experience/phase1_token_segmentation                  │
│  Inference: experience_phase1_segment/                                        │
│  Artifact:  8_experience_segments.json                            │
│  Labels:    O | B-SEG | I-SEG                                     │
│  Question:  "Where does one phrase end and the next begin?"       │
│  UI eval:   entryDividerLines (job head lines vs experienceEntryHeads) │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 10 — CLASSIFY (field type per phrase block)                 │
│  Training:  experience/phase3_segment_classification            │
│  Inference: experience_phase3_classify/                                        │
│  Artifact:  10_experience_classification.json                     │
│  Labels:    ROLE | COMP | DATE | DESC  (B-*/I-* per token)        │
│  Question:  "Is this phrase a role, company, date, or description?" │
└───────────────────────────────────────────────────────────────────┘
```

### Naming trap (memorize this)

| Inference folder | Inference step | Training folder name | What it actually does |
|------------------|----------------|----------------------|------------------------|
| `experience_phase1_segment/` | **8** | `phase1_token_segmentation` | Phrase **segment** |
| `experience_phase2_divider/` | **9** | `phase2_section_divider` | Entry **divide** |
| `experience_phase3_classify/` | **10** | `phase3_segment_classification` | Field **classify** |

Inference folder names now match training phase numbers (`experience_phase1_segment`, `experience_phase2_divider`, `experience_phase3_classify`). Deprecated API aliases: `experience_p1` → phase 2 divider, `experience_p2` → phase 1 segment, `experience_p3` → phase 3 classify.

---

## 2. Code isolation rules (mandatory)

From project `.cursorrules` and `CLAUDE.md`:

1. **One stage per file responsibility** — do not put boundary logic in `experience_phase3_classify`, segmentation logic in `experience_phase2_divider`, etc.
2. **Scripts live in the section subfolder** — `experience_phase2_divider/*.py`, `experience_phase1_segment/*.py`, `experience_phase3_classify/*.py`. Do not patch global `data/` or root dirs for one section’s behavior.
3. **Shared utilities stay abstract** — e.g. `overlay_mongo_labels.py`, `date_patterns.py` must not hardcode section-specific overrides or `if section == "EXPERIENCE"` hacks that only serve one stage.
4. **Cross-stage data passes through artifacts or token fields** — step 8 writes `bioLabel` (boundary); step 9 reads boundaries, writes `segLabel`; step 10 reads both, writes field `bioLabel`.
5. **Training parity** — when inference block-building diverges from training (`entry_block_dataset.py`), step 10 metrics and UI block text will drift. Mirror training heuristics in `experience_phase3_classify/data_utils.py`.

### What each module may read / write

| Module | Reads | Writes | Must NOT |
|--------|-------|--------|----------|
| `experience_phase2_divider` | section=EXPERIENCE tokens, Mongo `_fieldBioLabel` (hints only) | `bioLabel` = O/B-ENTRY/I-ENTRY | Emit B-SEG, ROLE, DATE; call step 9 heuristics that use label id 1 as B-SEG |
| `experience_phase1_segment` | step 8 boundaries on tokens, `_fieldBioLabel` (hints) | `segLabel` / `prediction` = O/B-SEG/I-SEG; `entryDividerLines` | Change boundary labels; slice entries at every scattered B-ENTRY token |
| `experience_phase3_classify` | step 8 boundaries, step 9 segLabel | field `bioLabel` = B-ROLE/B-COMP/…; `blockClassification` | Re-group entries using raw B-ENTRY on every token; rely on segLabel alone for block splits |

### Primary entry slice heads (shared concept, separate implementations)

**Problem:** Step 8 puts `B-ENTRY` on many tokens per job (role words, company, stray dates). Using *every* `B-ENTRY` line as an entry boundary creates 6–9 fake jobs instead of 3.

**Rule:** Group and evaluate jobs only at **primary slice heads** — lines with a job bullet (`•`) that also have boundary signal on the same line.

| Location | File |
|----------|------|
| Python (step 9, 10) | `experience_phase1_segment/entry_slice_heads.py` → `resolve_entry_slice_heads()` |
| TypeScript (UI) | `app/app/inference-v2/lib/entrySliceHeads.ts` |

**Exclude from slice heads:** date-only lines `( July'22 … )`, company continuation without bullet, sub-bullets `◦`.

---

## 3. Per-stage implementation map

### Step 9 — `experience_phase2_divider/`

| File | Role |
|------|------|
| `__init__.py` | Model inference + post-process pipeline |
| `entry_postprocess.py` | Promote/suppress B-ENTRY lines; `demote_boundary_on_date_tokens()` |
| `entry_style_heuristic.py` | Multi-style entry heads |
| `entry_span_expand.py` | Expand head lines to I-ENTRY spans until next head |
| `date_patterns.py` | `is_date_token()`, Unicode apostrophe year patterns |
| `heads_loader.py` | Load `experienceEntryHeads` from Mongo (hints / eval) |

**Post-process order (do not reorder casually):**

```
model logits
  → apply_entry_boundary_postprocess
  → apply_style_entry_heuristic
  → expand_entry_span_labels
  → demote_boundary_on_date_tokens
```

### Step 8 — `experience_phase1_segment/`

| File | Role |
|------|------|
| `__init__.py` | Per-entry phrase segmenter model + post-process |
| `entry_slice_heads.py` | Primary job head lines for entry grouping |
| `entry_divider_lines.py` | Build `entryDividerLines` report (matched / extra) |
| `gap_heuristic.py` | Column gaps, date tails, entity B-SEG promotion (**B-SEG=1 only here**) |

**Post-process:** `apply_segment_postprocess` = gap boundaries + entity/date heuristics.  
**Only promotes header entity types** (`ROLE`, `COMP`, `SDATE`, `EDATE`, …) — not `B-DESC` or boundary `B-ENTRY`.

### Step 10 — `experience_phase3_classify/`

| File | Role |
|------|------|
| `__init__.py` | Group entries → split blocks → classify → write field labels |
| `data_utils.py` | `split_entry_blocks()`, `merge_adjacent_date_blocks()`, `group_experience_entries()` |
| `block_classification_report.py` | Block-level GT vs pred table |

**Block split order (`split_entry_blocks`):**

1. `B-SEG` starts a new block **unless** `_should_skip_b_seg_split` (date continuation on same line, e.g. `Present` after `March 2023`)
2. Split at `(` when followed by a date token (`Pvt. Ltd. (` | `March 2023 Present )`)
3. Line boundary, delimiter, spatial gap (same as training `entry_block_dataset.py`)

**After split:** `merge_adjacent_date_blocks()` merges `(`, `July'22 -`, `March' 2023 )` fragments into one DATE block.

---

## 4. Label ID collision (critical)

Step 9 `gap_heuristic.py` uses integer labels:

```python
B_SEG = 1
I_SEG = 2
```

Step 8 boundaries use:

```python
O = 0, B-ENTRY = 1, I-ENTRY = 2
```

**Never call `apply_entity_and_date_heuristics` from step 8.** It was removed after it promoted Mongo `B-EDATE` / `B-SDATE` hints to label id `1`, which step 8 interpreted as `B-ENTRY` — so `Present`, `July'22`, `March'` became false entry boundaries.

If you need date-aware boundary logic, add it in `experience_phase2_divider/` with explicit `B-ENTRY`/`I-ENTRY` strings, not step 9 integer ids.

---

## 5. Solved failure catalog (do not re-break)

### 5.1 Step 8 — Unicode apostrophe dates not detected

| | |
|--|--|
| **Symptom** | `(July'22` not treated as date anchor; wrong boundary promotion |
| **Root cause** | ASCII-only `'` in regex; PDF uses `'` U+2019 |
| **Fix** | `experience_phase2_divider/date_patterns.py` — `APOSTROPHE_YEAR_RE`, `MONTH_YEAR_RE`, `is_date_token()` |
| **Verify** | `isDateToken: true` on artifact tokens; `dateTokenCount` in step 8 JSON |

### 5.2 Step 8 — `Present` / `July'22` tagged `B-ENTRY`

| | |
|--|--|
| **Symptom** | Date tokens get `B-ENTRY` in `9_experience_boundaries.json`; step 8/10 cascade breaks |
| **Root cause** | `apply_entity_and_date_heuristics` called from `experience_phase2_divider` (label id collision, §4) |
| **Fix** | Removed from step 8; added `demote_boundary_on_date_tokens()` in `entry_postprocess.py` |
| **Verify** | L15 `Present` → `I-ENTRY`; L34 `July'22` → `I-ENTRY` |

### 5.3 Step 9 — Too many entry blocks / extra divider lines

| | |
|--|--|
| **Symptom** | 9 entry blocks vs 3 GT jobs; entry divider shows 3 matched + 3 extra |
| **Root cause** | Slicing on every line with any `B-ENTRY` token; `gap_heuristic` promoted `B-DESC` / boundary labels to `B-SEG` |
| **Fix** | `entry_slice_heads.py`; slice heads = `•` line + boundary signal; gap heuristic limits to `HEADER_ENTITY_TYPES` |
| **Verify** | `entryDividerLines.metrics`: `matched: 3, extra: 0` on Karan |

### 5.4 Step 9 — `B-SEG` explosion (~81 → ~11 on Karan)

| | |
|--|--|
| **Symptom** | Dense bogus `B-SEG` on every entity token |
| **Root cause** | Same as §5.3 — promoting all `B-*` including `B-DESC`, `B-ENTRY` |
| **Fix** | `HEADER_ENTITY_TYPES` filter in `gap_heuristic.py` |
| **Verify** | Count `B-SEG` in `8_experience_segments.json` tokens |

### 5.5 Step 10 UI — 4 jobs in Structured View instead of 3

| | |
|--|--|
| **Symptom** | Structured View shows 4 jobs |
| **Root cause** | `structureResume.ts` flushed new job on every `ROLE` token; `Present )` misclassified as ROLE |
| **Fix** | Split only on primary role lines (`•` bullets); `experience_phase3_classify` uses `resolve_entry_slice_heads()` for grouping |
| **Verify** | 3 job keys: `JOB p1 L14`, `L26`, `L33` |

### 5.6 Step 10 — Two visual lines merged in block `text`

| | |
|--|--|
| **Symptom** | `Present ) ◦ Project…` or `July'22 ) ◦ Engaged…` as one block row |
| **Root cause** | `split_entry_blocks` split only on `B-SEG`; step 9 `I-SEG` spanned date line + next `◦` line |
| **Fix** | Also split on **line boundaries** (and delimiter/spatial heuristics) even when `segLabel` present |
| **Verify** | Separate block rows for date tail and `◦` project line |

### 5.7 Step 10 — Date blocks split or mislabeled

| | |
|--|--|
| **Symptom** | `Present )` as ROLE; `March 2023` as COMP; fragmented `( July'22 -` / `March' 2023 )` |
| **Root cause** | (a) False `B-SEG` on `Present` from step 9; (b) no split at `(` before date; (c) no merge of date fragments |
| **Fix** | `_should_skip_b_seg_split`, `_starts_date_after_open_paren`, `_is_date_fragment_block` + `merge_adjacent_date_blocks` in `data_utils.py` |
| **Verify** | L14: `March 2023 Present )` → DATE; L26: `( July'22 - March' 2023 )` → DATE; L33: `( Jan'22 - July'22 )` → DATE |

### 5.8 GT scoring — `Pvt. Ltd. (` shows gt=DATE, pred=COMP

| | |
|--|--|
| **Symptom** | Block row ❌ but prediction matches training report (COMP) |
| **Root cause** | Mongo token `(March` (one box) vs PDF `(` + `March` (two boxes); coord match assigns `B-SDATE` to `(` block |
| **Action** | **Not an inference bug.** Document when interpreting metrics. See README § Split vs merged tokens. |
| **Training GT** | `phase3_segment_classification/reports/minilm/per_resume/Karan.md`: COMP = `Pvt. Ltd. (`, DATE = `March 2023 Present )` |

---

## 6. Karan reference checklist (regression)

After any change to steps 8–10, rerun on Karan and confirm:

### Step 8 (`8_experience_segments.json`)

- [ ] 3 primary entry head lines: p1 L14, L26, L33 (via slice heads / entry heads)
- [ ] L15 `Present` → `I-ENTRY` (not `B-ENTRY`)
- [ ] L27/L34 date tokens → `I-ENTRY` only
- [ ] `dateTokenCount` > 0; apostrophe years flagged `isDateToken`

### Step 9 (`9_experience_boundaries.json`)

- [ ] `entryDividerLines`: 3 matched, 0 extra
- [ ] ~3 entry blocks in divider (not 6–9)
- [ ] `B-SEG` count modest (~10–15, not 80+)

### Step 10 (`10_experience_classification.json`)

- [ ] 3 entry keys in block table
- [ ] L14 DATE block: `March 2023 Present )` (not merged with `◦ Project`)
- [ ] L26 DATE: `( July'22 - March' 2023 )` single block
- [ ] L33 DATE: `( Jan'22 - July'22 )` single block
- [ ] No `Present ) ◦ …` combined text rows

### Rerun commands

```bash
curl -X POST http://localhost:8000/inference-v2/runs/Karan_<slug>/rerun/experience_phase2_divider
curl -X POST http://localhost:8000/inference-v2/runs/Karan_<slug>/rerun/experience_phase1_segment
curl -X POST http://localhost:8000/inference-v2/runs/Karan_<slug>/rerun/experience_phase3_classify
```

Step 10 rerun replays steps 8+9 artifacts from disk — **rerun step 8 first** if boundary logic changed.

## 6.1 Step 10 locked baseline

Step 10 is **LOCKED** (Karan regression passes with 96.3% macro F1 proxy; date blocks correct). Do not modify without re-open.

*   **Artifact:** `10_experience_classification.json`
*   **Module:** `experience_phase3_classify/` only
*   **Regression resume:** Karan (`Karan_9e7bb9` or latest slug)
*   **Must hold:** 3 jobs (L14/L26/L33); DATE blocks `March 2023 Present )`, `( July'22 - March' 2023 )`, `( Jan'22 - July'22 )`; no date+project line merges.
*   **Known scoring quirk:** `Pvt. Ltd. (` gt=DATE vs pred=COMP — Mongo `(March` vs PDF `(`+`March` tokenization (documented in guide §5.8); not a model bug.

No new `step10FieldGt.ts` needed — step 10 GT already uses raw Mongo `bioLabel` per block (see README § Step 9 & 10).

---


## 7. Debugging workflow

1. **Identify the stage** — boundary vs phrase vs field? Do not fix step 10 if step 8 boundaries are wrong.
2. **Print token traces** at page/line for Karan L14–L35 (dates + bullets).
3. **Compare to training reports:**
   - Step 8/9 line eval: `training_pipeline/experience/phase2_section_divider/reports/per_resume/Karan.md`
   - Step 9 phrases: `training_pipeline/experience/phase1_token_segmentation/reports/…`
   - Step 10 blocks: `training_pipeline/experience/phase3_segment_classification/reports/minilm/per_resume/Karan.md`
4. **Check slice heads** — `resolve_entry_slice_heads()` output before grouping.
5. **Check block text** — `split_entry_blocks` → `merge_adjacent_date_blocks` before classifier.
6. **GT vs pred** — coordinate key `(page, round(x0,2), round(y0,2))`; expect split-token GT quirks (§5.8).

### CLI trace template

```bash
cd training-engine
PYTHONPATH=. python3 -c "
import json
p8 = json.load(open('inference_runs/Karan_<slug>/8_experience_segments.json'))
p9 = json.load(open('inference_runs/Karan_<slug>/9_experience_boundaries.json'))
p10 = json.load(open('inference_runs/Karan_<slug>/10_experience_classification.json'))
# print per-line boundary / seg / field for L14-L35
"
```

---

## 8. UI ↔ Python parity

Keep these in sync when changing entry-head logic:

| Concern | Python | TypeScript |
|---------|--------|------------|
| Primary slice heads | `experience_phase1_segment/entry_slice_heads.py` | `app/.../lib/entrySliceHeads.ts` |
| Entry divider report | `experience_phase1_segment/entry_divider_lines.py` | `app/.../lib/entryDividerLines.ts` |
| Structured job count | `experience_phase3_classify` grouping | `app/.../lib/structureResume.ts` |
| Step 9 boundary GT | `scratch/export_mongo_boundary_gt.py` | `app/.../lib/step8BoundaryGt.ts` |

---

## 9. Related files (quick index)

```
training-engine/inference_v2/
├── EXPERIENCE_PIPELINE_GUIDE.md   ← this document
├── README.md                      ← artifacts & GT
├── experience_phase2_divider/                 ← step 8 divide
├── experience_phase1_segment/                 ← step 9 segment
│   ├── entry_slice_heads.py       ← job grouping (critical)
│   └── gap_heuristic.py           ← B-SEG only; never import in p1
├── experience_phase3_classify/                 ← step 10 classify
│   └── data_utils.py              ← block split/merge (critical)
└── overlay_mongo_labels.py        ← _fieldBioLabel overlay (hints)

training_pipeline/experience/
├── phase2_section_divider/        ← training for step 8
├── phase1_token_segmentation/     ← training for step 9
└── phase3_segment_classification/
    └── entry_block_dataset.py     ← block split ground truth for step 10
```

---

*Last updated from inference-v2 debugging session (Karan): boundary date promotion, entry slice heads, block line-merge, date fragment merge.*
