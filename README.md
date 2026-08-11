# Parselex

**Turn a resume PDF into clean, structured JSON — no LLM, no API key, no per-resume cost.**

Most "AI resume parsers" are a thin wrapper around a GPT prompt: slow, non-deterministic, and billed per call. Parselex is the opposite bet — a chain of small, purpose-built PyTorch classifiers (a few MB each, MiniLM-backbone) that runs entirely on your own CPU, deterministically, offline, for free. Point it at a PDF and get back a fully structured resume: personal info, a summary, work history grouped by job, projects, and a skills list — each field individually labeled with a confidence score, not a best-effort blob of text.

**How it works:** the PDF is decomposed token-by-token (position, font, style) and pushed through a 13-stage pipeline — section detection → entry boundary detection → field classification, run separately per section (education / experience / projects / skills / personal). Each stage is its own small model plus a layer of deterministic post-processing heuristics that catch the kind of boundary/continuity mistakes classifiers make (a bullet point wrongly split across a comma, a company name misfiled as a description, that sort of thing). The result is intentionally boring and predictable — same input, same output, every time.

Two ways to use it:
- **API only** — one HTTP call to a FastAPI backend, get structured JSON back. No UI required, no external dependencies beyond the model weights.
- **UI** — a Next.js app to upload a resume and watch each of the 13 pipeline stages run in real time, inspect the raw output of every stage, and toggle between fp32/int8 model precision.

Both talk to the same engine — the UI is just a client of the API. Everything here is inference-only: no training code, no training data, nothing that requires a database. Clone it, download the weights, run it.

## Status & limitations

This is an early-stage project, not a polished drop-in product yet — read this before wiring it into a real workflow.

- **Small training set.** The classifiers were trained on a limited number of labeled resumes. Expect solid accuracy on resumes that look broadly like the training distribution (standard Western tech-resume conventions) and more misclassifications the further a resume drifts from that — unusual layouts, heavy formatting, non-English content, etc. Don't treat the output as ground truth without a review step in your workflow.
- **Single-column resumes only.** The pipeline's reading-order logic is not built for multi-column layouts (side-by-side sidebar + main content, etc.). Feed it a two-column resume and expect scrambled section/field assignment, not a clean error.
- **No overall resume score.** Parselex extracts and labels fields — it does not produce any kind of quality score, ATS-match score, or ranking. If your workflow needs a "how good is this resume" number, that's out of scope here; you'd build that on top of the structured output yourself.
- **Confidence scores help, but aren't a substitute for validation.** Every field comes with a model confidence, useful for flagging low-confidence extractions for human review — but a high-confidence field can still be wrong (see the small-training-set point above).

If you hit a resume that parses badly, the most useful thing you can do is open an issue with the (anonymized/redacted, if needed) PDF — the training set is the actual bottleneck here, not the pipeline architecture.

## Structure

```
parselex/
  engine/            FastAPI + PyTorch inference pipeline (the actual work happens here)
  engine/parity/     parity helper modules a few classifiers dynamically import (spatial features, label rules) — inference-only, no training scripts
  web/               Next.js UI — optional, purely a client of engine/
  model_weights/     .pt checkpoints, not committed to git — see model_weights/README.md
  full-database/     Karan.pdf, the bundled demo resume
  examples/          minimal client scripts (Python + Node) calling the API directly
```

## Setup

### 1. Download model weights
Required either way (UI or API-only) — the engine won't start correctly without them. Hosted on Hugging Face: [`karan963/parselex-weights`](https://huggingface.co/karan963/parselex-weights). Follow `model_weights/README.md` to fetch and place files into `model_weights/<stage>/`. ~2.5GB total (fp32 + int8 checkpoints for 13 stages).

### 2. Run the engine
```bash
cd engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Verify it's up:
```bash
curl localhost:8000/health   # {"status":"ok"}
```
This is all you need for API-only usage — skip to [API](#api) below.

### 3. (Optional) Run the web UI
```bash
cd web
npm install
cp .env.local.example .env.local
npm run dev
```
Open `http://localhost:3000` — redirects to `/inference-v2`. Upload a resume PDF, or click "run default" to try the bundled demo resume. `FASTAPI_URL` in `.env.local` points the UI at the engine (defaults to `localhost:8000`).

The UI is useful for debugging/inspecting individual pipeline stages (headings, boundaries, per-field classification, artifact diffing) — if you just want structured JSON out, the API alone is enough.

There's also a `NEXT_PUBLIC_ENABLE_GT` flag (off by default) that toggles a ground-truth/MongoDB comparison mode used during model development — irrelevant unless you're working on the classifiers themselves and have a MongoDB-backed labeling app running separately. Leave it off.

## API

### One-shot parse (recommended)

Upload a PDF, block until the whole pipeline finishes, get the final structured entities back in one response.

```bash
curl -X POST "http://localhost:8000/inference-v2/parse" \
  -F "file=@resume.pdf"
```

Optional `?precision=fp32|int8` query param (default `fp32`; `int8` is faster/smaller, slightly lower accuracy).

Response:
```json
{
  "slug": "resume_ab12cd",
  "resumeId": "resume",
  "structured": {
    "SECTION_HEADINGS": ["EXPERIENCE", "EDUCATION", "..."],
    "PERSONAL": [{ "label": "NAME", "value": "..." }, { "label": "EMAIL", "value": "..." }],
    "SUMMARY": "One-paragraph professional summary, heuristically extracted — no model involved.",
    "EDUCATION": [{ "label": "DEG", "value": "..." }, { "label": "INST", "value": "..." }],
    "EDUCATION_ENTRIES": [[ /* fields grouped per education entry */ ]],
    "EXPERIENCE": [{ "label": "ROLE", "value": "..." }, { "label": "COMP", "value": "..." }],
    "EXPERIENCE_ENTRIES": [[ /* fields grouped per job entry */ ]],
    "PROJECTS": [{ "label": "PROJ_NAME", "value": "..." }, { "label": "DESC", "value": "..." }],
    "PROJECT_ENTRIES": [[ /* fields grouped per project entry */ ]],
    "SKILLS": ["Python", "React", "..."]
  }
}
```

`*_ENTRIES` arrays group the flat field lists above them into per-entry chunks (one array per job/degree/project) — use those if you need entry boundaries; use the flat `PERSONAL`/`EDUCATION`/`EXPERIENCE`/`PROJECTS` arrays if you just want every labeled field.

Takes ~5–10s per resume on CPU (first request after startup is slower — models load lazily per stage and stay cached in memory after).

### Async flow (what the UI uses)

If you want progress visibility per stage instead of one blocking call:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/inference-v2/run` | Upload a PDF, starts a background run, returns `{slug, status: "running"}` immediately |
| `POST` | `/inference-v2/run/default` | Same, but runs the bundled demo resume |
| `GET` | `/inference-v2/runs` | List all runs |
| `GET` | `/inference-v2/runs/{slug}` | Poll run status/manifest — `status` becomes `"completed"` or `"failed"`, `artifacts` lists per-stage JSON files as they're written |
| `GET` | `/inference-v2/runs/{slug}/pdf` | Fetch the original uploaded PDF |
| `GET` | `/inference-v2/runs/{slug}/artifacts/{filename}` | Fetch a specific stage's raw output, e.g. `8_experience_segments.json` |

Poll `GET /inference-v2/runs/{slug}` until `status` is `completed`, then read `structured.json` via the artifacts endpoint for the same final output `/parse` returns directly.

(There are also `POST /inference-v2/runs/{slug}/rerun/{stage}` and `/{stage}/predict` endpoints per pipeline stage — internal, used by the UI's per-stage debug/rerun tools. Not a stable public contract; use `/parse` or `/run` instead.)

### Through the Next.js proxy

If running the web UI, the same `/parse` and `/run` endpoints are also reachable via Next.js API routes at `http://localhost:3000/api/inference-v2/...` (they just forward to the FastAPI engine using `FASTAPI_URL`). Useful if you're calling from browser JS and want to avoid CORS, or want one origin for both UI and API.

```bash
curl -X POST "http://localhost:3000/api/inference-v2/parse" -F "file=@resume.pdf"
```

### Example client scripts

`examples/` has minimal, dependency-free clients (stdlib / `fetch` only) that call `/inference-v2/parse`:

```bash
python3 examples/parse_resume.py                          # parses bundled demo resume, saves examples/output/Karan.json
python3 examples/parse_resume.py resume.pdf --out result.json
node examples/parse_resume.mjs resume.pdf --url http://localhost:8000 --precision int8
```

## Pipeline stages

Each resume passes through, in order: token extraction (`pdfplumber`) → section heading detection → section assignment → then per-section (education / experience / project): token segmentation → boundary/divider → field classification. Skills and personal info are classified directly without a boundary stage. See `engine/inference_v2/config.py:PIPELINE_STEPS` for the exact stage list and artifact filenames.

## Troubleshooting

- **Engine fails to import with `ModuleNotFoundError`** — a model's checkpoint is missing. Confirm `model_weights/<stage>/` is populated (see step 1).
- **`/parse` times out** — first request per stage loads that stage's model into memory; subsequent requests are fast. If it never returns, check `uvicorn` logs for a stage-level traceback.
- **UI shows errors but engine is healthy** — check `web/.env.local` has the right `FASTAPI_URL`, and that the engine is actually reachable from wherever `npm run dev` is running.

## License

MIT — see [`LICENSE`](./LICENSE). Covers this repository's code only; the accompanying paper is
licensed separately (CC BY 4.0).
