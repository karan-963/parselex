# Model weights

Not checked into git (too large for GitHub) — hosted as a single zip on Cloudflare storage.

**Download URL:** `https://pub-6d31e47e899b48e69c7e7a45f99b3565.r2.dev/parselex-model-weights.zip`
**MD5:** `1669fbcbf3bf772648f3fa4b5efd68d2`
**SHA256:** `2fe9f5f52b8042d6e5e5c34f32e351b1ceb3a9bdc9119db71a4788e234466907`
**Size:** ~2.0GB zipped (~2.5GB extracted)

## Quick setup (recommended)

```bash
cd model_weights
./download.sh
# or, cross-platform:
python3 download.py
```

Both scripts download the zip, verify it against the MD5 above, extract into `model_weights/`, and delete the zip. Edit the `URL`/`MD5_EXPECTED` constants at the top of the script once (or set `PARSELEX_WEIGHTS_URL` env var) before running.

## Manual setup

1. Download the zip from the URL above.
2. Verify integrity:
   ```bash
   # macOS
   md5 -q parselex-model-weights.zip   # compare to MD5 above
   # Linux
   md5sum parselex-model-weights.zip   # compare to MD5 above
   ```
3. Extract into this folder: `unzip parselex-model-weights.zip -d .`

## Layout

Each pipeline stage gets its own subfolder here. `engine/inference_v2/<stage>/*.pt` are symlinks pointing at these files — extracting the zip here is all that's needed, no code changes.

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

`best_model.pt` = fp32 checkpoint. `best_model_int8.pt` = quantized (smaller, faster). The UI's precision selector (`fp32` / `int8`) picks between them per-request — the zip ships both.
