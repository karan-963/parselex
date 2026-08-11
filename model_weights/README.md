# Model weights

Not checked into git (too large for GitHub) — hosted on Hugging Face:
[`karan963/parselex-weights`](https://huggingface.co/karan963/parselex-weights). ~2.5GB total
(fp32 + int8 checkpoints for 13 stages).

## Quick setup (recommended)

```bash
pip install huggingface_hub   # if not already installed
cd model_weights
./download.sh
# or, cross-platform:
python3 download.py
```

Both scripts pull straight from the Hugging Face repo above into `model_weights/<stage>/` — no
zip, no manual integrity check (Hugging Face's own transfer already handles that). Set
`PARSELEX_WEIGHTS_REPO` (shell) / `PARSELEX_WEIGHTS_REPO` env var (Python) to point at a fork or
mirror instead.

## Manual setup

```bash
hf download karan963/parselex-weights --local-dir model_weights --include "*/*.pt"
# or, via Python:
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='karan963/parselex-weights', local_dir='model_weights', allow_patterns=['*/*.pt'])"
```

## Layout

Each pipeline stage gets its own subfolder here. `engine/inference_v2/<stage>/*.pt` are symlinks pointing at these files — downloading into this folder is all that's needed, no code changes.

| Folder | Files needed |
|---|---|
| `section_p1/` | `best_model_line_minilm.pt`, `best_model_line_minilm_int8.pt` |
| `section_p2/` | `best_model.pt`, `best_model_int8.pt` |
| `education_phase1_segment/` | `best_model.pt`, `best_model_int8.pt` |
| `education_phase2_divider/` | `best_model.pt`, `best_model_int8.pt` |
| `education_phase3_classify/` | `best_model.pt`, `best_model_int8.pt` |
| `experience_phase1_segment/` | `best_model.pt`, `best_model_int8.pt` |
| `experience_phase2_divider/` | `best_model.pt`, `best_model_int8.pt` |
| `experience_phase3_classify/` | `best_model.pt`, `best_model_int8.pt` |
| `project_phase1_segment/` | `best_model.pt`, `best_model_int8.pt` |
| `project_phase2_divider/` | `best_model.pt`, `best_model_int8.pt` |
| `project_phase3_classify/` | `best_model.pt`, `best_model_int8.pt` |
| `skills_classify/` | `best_model.pt`, `best_model_int8.pt` |
| `personal_classify/` | `best_model.pt`, `best_model_int8.pt` |

`best_model.pt` = fp32 checkpoint. `best_model_int8.pt` = quantized (smaller, faster). The UI's precision selector (`fp32` / `int8`) picks between them per-request — the HF repo ships both.
