"""FastAPI routes for Inference V2."""

from __future__ import annotations

import os
import shutil
import threading
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from . import config, storage
from .model_precision import get_precision, set_precision
from .pipeline import run_pipeline


async def _apply_request_precision(precision: str = Query("fp32")) -> str:
    """Router dependency: set the active model precision from the ``?precision`` query.

    Runs in the request context so the value propagates to synchronous stage
    handlers (reruns/predict). Background pipeline runs receive it explicitly.
    """
    return set_precision(precision)


router = APIRouter(
    prefix="/inference-v2",
    tags=["inference-v2"],
    dependencies=[Depends(_apply_request_precision)],
)

EXPERIENCE_BOUNDARIES_ARTIFACT = "9_experience_boundaries.json"
EXPERIENCE_BOUNDARIES_LEGACY = "8_experience_boundaries.json"
EXPERIENCE_SEGMENTS_ARTIFACT = "8_experience_segments.json"
EXPERIENCE_SEGMENTS_LEGACY = "9_experience_segments.json"
EXPERIENCE_CLASSIFY_ARTIFACT = "10_experience_classification.json"
EXPERIENCE_CLASSIFY_LEGACY = "10_experience_fields.json"


def _read_artifact_legacy(slug: str, primary: str, *legacy: str) -> dict[str, Any]:
    try:
        return storage.read_json(slug, primary)
    except FileNotFoundError:
        for name in legacy:
            try:
                return storage.read_json(slug, name)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(primary)


def _start_background(slug: str, pdf_path: str, *, cleanup: bool = False, precision: str = "fp32") -> None:
    def _worker() -> None:
        try:
            run_pipeline(slug, pdf_path, precision=precision)
        finally:
            if cleanup and os.path.isfile(pdf_path):
                os.remove(pdf_path)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


@router.post("/run")
async def run_inference(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF file required")

    precision = get_precision()
    slug = storage.generate_slug(file.filename)
    tmp_dir = os.path.join(config.RUNS_DIR, "_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.pdf")

    with open(tmp_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    storage.create_run(slug, source_pdf=tmp_path, original_filename=file.filename, precision=precision)
    _start_background(slug, tmp_path, cleanup=True, precision=precision)
    return {"slug": slug, "status": "running", "modelPrecision": precision}


@router.post("/run/default")
def run_default() -> dict[str, Any]:
    if not os.path.isfile(config.DEFAULT_PDF):
        raise HTTPException(status_code=404, detail=f"Default PDF not found: {config.DEFAULT_PDF}")

    precision = get_precision()
    slug = storage.generate_slug("Karan.pdf")
    storage.create_run(slug, source_pdf=config.DEFAULT_PDF, original_filename="Karan.pdf", precision=precision)
    _start_background(slug, config.DEFAULT_PDF, precision=precision)
    return {"slug": slug, "status": "running", "source": config.DEFAULT_PDF, "modelPrecision": precision}


@router.post("/parse")
async def parse_resume(file: UploadFile = File(...)) -> dict[str, Any]:
    """Synchronous one-shot endpoint: upload a PDF, get structured JSON back directly.

    Unlike /run (which starts a background run you poll via /runs/{slug}), this
    blocks until the pipeline finishes and returns the final structured entities.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF file required")

    precision = get_precision()
    slug = storage.generate_slug(file.filename)
    tmp_dir = os.path.join(config.RUNS_DIR, "_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.pdf")

    with open(tmp_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    storage.create_run(slug, source_pdf=tmp_path, original_filename=file.filename, precision=precision)
    try:
        result = run_pipeline(slug, tmp_path, precision=precision)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)

    return {
        "slug": result["slug"],
        "resumeId": result["resumeId"],
        "structured": result["structured"],
    }


@router.get("/runs")
def list_runs() -> dict[str, Any]:
    return {"runs": storage.list_runs()}


@router.get("/runs/{slug}")
def get_run(slug: str) -> dict[str, Any]:
    manifest = storage.read_manifest(slug)
    if not manifest:
        raise HTTPException(status_code=404, detail="Run not found")

    payload: dict[str, Any] = dict(manifest)
    if manifest.get("status") == "completed":
        try:
            structured = storage.read_json(slug, "structured.json")
            payload["structured"] = structured.get("entities", structured)
            payload["resumeId"] = structured.get("resumeId")
        except FileNotFoundError:
            pass
    return payload


@router.get("/runs/{slug}/pdf")
def get_run_pdf(slug: str) -> FileResponse:
    path = storage.artifact_path(slug, "input.pdf")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path, media_type="application/pdf", filename=f"{slug}.pdf")


@router.get("/runs/{slug}/artifacts/{filename}")
def get_artifact(slug: str, filename: str) -> FileResponse:
    path = storage.artifact_path(slug, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    media = "application/json" if filename.endswith(".json") else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=filename)


from pydantic import BaseModel

class TokensRequest(BaseModel):
    tokens: list[dict[str, Any]]


@router.post("/runs/{slug}/rerun/section_p1")
def rerun_section_p1(slug: str) -> dict[str, Any]:
    try:
        data = storage.read_json(slug, "1_extracted_tokens.json")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Extracted tokens not found. Run base inference first.")
    
    tokens = data["tokens"]
    from .section_p1 import run_section_phase1
    p1_out = run_section_phase1(tokens)
    storage.write_json(slug, "2_section_headings.json", p1_out)
    return {
        "status": "completed",
        "headingsCount": p1_out["headingCount"],
        "headings": p1_out["headings"]
    }


@router.post("/section_p1/predict")
def predict_section_p1(req: TokensRequest) -> dict[str, Any]:
    from .section_p1 import run_section_phase1
    tokens = [dict(t) for t in req.tokens]
    return run_section_phase1(tokens)


@router.post("/runs/{slug}/rerun/section_p2")
def rerun_section_p2(slug: str) -> dict[str, Any]:
    # Phase 2 requires bioLabels to be present on the tokens.
    # Therefore, we must read them from 2_section_headings.json or reconstruct them,
    # or just read from the last successful stage's tokens if available.
    # In run_pipeline, tokens are processed in-place.
    # For a clean rerun of section_p2, we load 1_extracted_tokens.json, apply 2_section_headings.json headings,
    # and then run section_p2.
    try:
        raw_data = storage.read_json(slug, "1_extracted_tokens.json")
        p1_data = storage.read_json(slug, "2_section_headings.json")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Required artifacts (1_extracted_tokens or 2_section_headings) not found.")

    tokens = raw_data["tokens"]
    
    # Apply heading predictions from p1
    from .section_p1 import apply_heading_predictions
    from .section_p1.line_builder import build_parser_lines
    
    # We can reconstruct predictions list from p1_data headings
    # Or run apply_heading_predictions using predictions proxies
    class PredProxy:
        def __init__(self, page, lineIndex):
            self.key = (page, lineIndex)
            self.is_heading = True
            
    proxies = [PredProxy(h["page"], h["lineIndex"]) for h in p1_data.get("headings", [])]
    apply_heading_predictions(tokens, proxies)

    from .section_p2 import run_section_phase2
    p2_out = run_section_phase2(tokens)
    storage.write_json(slug, "3_section_labels.json", p2_out)
    return {
        "status": "completed",
        "chunkCount": p2_out["chunkCount"],
        "chunks": p2_out["chunks"]
    }


@router.post("/section_p2/predict")
def predict_section_p2(req: TokensRequest) -> dict[str, Any]:
    from .section_p2 import run_section_phase2
    tokens = [dict(t) for t in req.tokens]
    return run_section_phase2(tokens)


def _prepare_tokens_after_section_labels(slug: str) -> tuple[list[dict], str]:
    """Replay steps 1–3 token prep (headings, sections, original lineIndex restore)."""
    raw_data = storage.read_json(slug, "1_extracted_tokens.json")
    p1_data = storage.read_json(slug, "2_section_headings.json")
    tokens = raw_data["tokens"]

    from .section_p1 import apply_heading_predictions

    class PredProxy:
        def __init__(self, page, lineIndex):
            self.key = (page, lineIndex)
            self.is_heading = True

    proxies = [PredProxy(h["page"], h["lineIndex"]) for h in p1_data.get("headings", [])]
    apply_heading_predictions(tokens, proxies)

    from .section_p2 import run_section_phase2
    run_section_phase2(tokens)

    original_tokens = raw_data["tokens"]
    orig_by_coord = {}
    for t in original_tokens:
        key = (t.get("page"), round(float(t.get("x0", 0)), 2), round(float(t.get("y0", 0)), 2))
        orig_by_coord[key] = t.get("lineIndex")

    for t in tokens:
        key = (t.get("page"), round(float(t.get("x0", 0)), 2), round(float(t.get("y0", 0)), 2))
        if key in orig_by_coord:
            t["lineIndex"] = orig_by_coord[key]

    from .section_p1.data_utils import sort_tokens_by_reading_order
    tokens[:] = sort_tokens_by_reading_order(tokens)
    return tokens, raw_data.get("resumeId") or slug


@router.post("/runs/{slug}/rerun/education_phase2_divider")
def rerun_education_phase2_divider(slug: str) -> dict[str, Any]:
    """Re-run education training phase 2 (entry boundary divider)."""
    try:
        tokens, resume_id = _prepare_tokens_after_section_labels(slug)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Required preceding artifacts not found.")

    from .education_phase2_divider import run_education_phase2_divider
    edu_out = run_education_phase2_divider(tokens, resume_id)
    storage.write_json(slug, "5_education_boundaries.json", edu_out)

    return {
        "status": "completed",
        "tokenCount": edu_out["tokenCount"],
        "nonOCount": edu_out["nonOCount"],
        "fbaPercent": edu_out.get("entryDividerLines", {}).get("metrics", {}).get("fbaPercent"),
    }


@router.post("/runs/{slug}/rerun/education_phase3_classify")
def rerun_education_phase3_classify(slug: str) -> dict[str, Any]:
    """Re-run education training phase 3 (segment field classification)."""
    try:
        tokens, resume_id = _prepare_tokens_after_section_labels(slug)
        storage.read_json(slug, "5_education_boundaries.json")
        storage.read_json(slug, "4_education_segments.json")
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Required preceding artifacts not found: {exc}",
        )

    boundary_data = storage.read_json(slug, "5_education_boundaries.json")
    segment_data = storage.read_json(slug, "4_education_segments.json")

    boundary_tok_map: dict[tuple, str] = {}
    for t in boundary_data.get("tokens", []):
        key = (t["page"], t["lineIndex"], t["tokenIndex"])
        boundary_tok_map[key] = t.get("prediction", "O")

    for t in tokens:
        key = (t.get("page"), t.get("lineIndex"), t.get("tokenIndex"))
        if key in boundary_tok_map:
            t["bioLabel"] = boundary_tok_map[key]
            t["bio_label"] = boundary_tok_map[key]

    segment_tok_map: dict[tuple, str] = {}
    for t in segment_data.get("tokens", []):
        key = (t["page"], t["lineIndex"], t["tokenIndex"])
        segment_tok_map[key] = t.get("prediction", "O")

    for t in tokens:
        key = (t.get("page"), t.get("lineIndex"), t.get("tokenIndex"))
        if key in segment_tok_map:
            t["segLabel"] = segment_tok_map[key]

    from .education_phase3_classify import run_education_phase3_classify
    edu_out = run_education_phase3_classify(tokens, resume_id)
    storage.write_json(slug, "6_education_fields.json", edu_out)

    return {
        "status": "completed",
        "tokenCount": edu_out["tokenCount"],
        "nonOCount": edu_out["nonOCount"],
        "segmentAccuracy": edu_out.get("blockClassification", {}).get("metrics", {}).get("segmentAccuracyPercent"),
    }


@router.post("/education_phase3_classify/predict")
def predict_education_phase3_classify(req: TokensRequest) -> dict[str, Any]:
    from .education_phase3_classify import run_education_phase3_classify
    tokens = [dict(t) for t in req.tokens]
    return run_education_phase3_classify(tokens)


@router.post("/runs/{slug}/rerun/skills_classify")
def rerun_skills_classify(slug: str) -> dict[str, Any]:
    """Re-run skills token BIO classification (step 7)."""
    try:
        tokens, resume_id = _prepare_tokens_after_section_labels(slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Required preceding artifacts not found: {exc}")

    from .skills_classify import run_skills_classify
    skills_out = run_skills_classify(tokens, resume_id)
    storage.write_json(slug, "7_skills_fields.json", skills_out)

    return {
        "status": "completed",
        "tokenCount": skills_out["tokenCount"],
        "nonOCount": skills_out["nonOCount"],
        "tokenAccuracy": skills_out.get("tokenClassification", {}).get("metrics", {}).get("tokenAccuracyPercent"),
    }


@router.post("/skills_classify/predict")
def predict_skills_classify(req: TokensRequest) -> dict[str, Any]:
    from .skills_classify import run_skills_classify
    tokens = [dict(t) for t in req.tokens]
    return run_skills_classify(tokens)


@router.post("/runs/{slug}/rerun/personal_classify")
def rerun_personal_classify(slug: str) -> dict[str, Any]:
    """Re-run personal segment field classification (step 15)."""
    try:
        tokens, resume_id = _prepare_tokens_after_section_labels(slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Required preceding artifacts not found: {exc}")

    from .personal_classify import run_personal_classify
    personal_out = run_personal_classify(tokens, resume_id)
    storage.write_json(slug, "15_personal_fields.json", personal_out)

    return {
        "status": "completed",
        "tokenCount": personal_out["tokenCount"],
        "nonOCount": personal_out["nonOCount"],
        "segmentAccuracy": personal_out.get("blockClassification", {}).get("metrics", {}).get("segmentAccuracyPercent"),
    }


@router.post("/personal_classify/predict")
def predict_personal_classify(req: TokensRequest) -> dict[str, Any]:
    from .personal_classify import run_personal_classify
    tokens = [dict(t) for t in req.tokens]
    return run_personal_classify(tokens)


@router.post("/runs/{slug}/rerun/education_phase1_segment")
def rerun_education_phase1_segment(slug: str) -> dict[str, Any]:
    """Re-run education training phase 1 (token segmentation)."""
    try:
        tokens, resume_id = _prepare_tokens_after_section_labels(slug)
        try:
            boundary_data = storage.read_json(slug, "5_education_boundaries.json")
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="5_education_boundaries.json not found. Run education_phase2_divider first.",
            )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Required preceding artifacts not found.")

    boundary_tok_map: dict[tuple, str] = {}
    for t in boundary_data.get("tokens", []):
        key = (t["page"], t["lineIndex"], t["tokenIndex"])
        boundary_tok_map[key] = t.get("prediction", "O")

    for t in tokens:
        key = (t.get("page"), t.get("lineIndex"), t.get("tokenIndex"))
        if key in boundary_tok_map:
            t["bioLabel"] = boundary_tok_map[key]
            t["bio_label"] = boundary_tok_map[key]

    from .education_phase1_segment import run_education_phase1_segment
    edu_out = run_education_phase1_segment(tokens, resume_id)
    storage.write_json(slug, "4_education_segments.json", edu_out)

    return {
        "status": "completed",
        "tokenCount": edu_out["tokenCount"],
        "nonOCount": edu_out["nonOCount"],
        "tokenAccuracy": edu_out.get("tokenSegmentation", {}).get("metrics", {}).get("tokenAccuracyPercent"),
    }


@router.post("/education_phase2_divider/predict")
def predict_education_phase2_divider(req: TokensRequest) -> dict[str, Any]:
    from .education_phase2_divider import run_education_phase2_divider
    tokens = [dict(t) for t in req.tokens]
    return run_education_phase2_divider(tokens)


@router.post("/education_phase1_segment/predict")
def predict_education_phase1_segment(req: TokensRequest) -> dict[str, Any]:
    from .education_phase1_segment import run_education_phase1_segment
    tokens = [dict(t) for t in req.tokens]
    return run_education_phase1_segment(tokens)


@router.post("/runs/{slug}/rerun/experience_phase2_divider")
def rerun_experience_phase2_divider(slug: str) -> dict[str, Any]:
    try:
        raw_data = storage.read_json(slug, "1_extracted_tokens.json")
        p1_data = storage.read_json(slug, "2_section_headings.json")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Required preceding artifacts not found.")

    tokens = raw_data["tokens"]

    from .section_p1 import apply_heading_predictions
    class PredProxy:
        def __init__(self, page, lineIndex):
            self.key = (page, lineIndex)
            self.is_heading = True

    proxies = [PredProxy(h["page"], h["lineIndex"]) for h in p1_data.get("headings", [])]
    apply_heading_predictions(tokens, proxies)

    from .section_p2 import run_section_phase2
    run_section_phase2(tokens)

    from .section_p1.data_utils import sort_tokens_by_reading_order
    tokens[:] = sort_tokens_by_reading_order(tokens)

    from .experience_phase2_divider import run_experience_phase2_divider
    resume_id = raw_data.get("resumeId") or slug
    exp_out = run_experience_phase2_divider(tokens, resume_id)
    storage.write_json(slug, EXPERIENCE_BOUNDARIES_ARTIFACT, exp_out)

    return {
        "status": "completed",
        "tokenCount": exp_out["tokenCount"],
        "nonOCount": exp_out["nonOCount"],
    }


@router.post("/runs/{slug}/rerun/experience_p1")
def rerun_experience_p1(slug: str) -> dict[str, Any]:
    """Deprecated alias — use /rerun/experience_phase2_divider."""
    return rerun_experience_phase2_divider(slug)


@router.post("/experience_phase2_divider/predict")
def predict_experience_phase2_divider(req: TokensRequest) -> dict[str, Any]:
    from .experience_phase2_divider import run_experience_phase2_divider
    tokens = [dict(t) for t in req.tokens]
    return run_experience_phase2_divider(tokens)


@router.post("/experience_p1/predict")
def predict_experience_p1(req: TokensRequest) -> dict[str, Any]:
    """Deprecated alias — use /experience_phase2_divider/predict."""
    return predict_experience_phase2_divider(req)


@router.post("/runs/{slug}/rerun/experience_phase1_segment")
def rerun_experience_phase1_segment(slug: str) -> dict[str, Any]:
    """Re-run experience training phase 1 (token segmentation)."""
    try:
        raw_data = storage.read_json(slug, "1_extracted_tokens.json")
        p1_data = storage.read_json(slug, "2_section_headings.json")
        try:
            p1_boundaries = _read_artifact_legacy(
                slug, EXPERIENCE_BOUNDARIES_ARTIFACT, EXPERIENCE_BOUNDARIES_LEGACY,
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"{EXPERIENCE_BOUNDARIES_ARTIFACT} not found. Run experience_phase2_divider first.",
            )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Required preceding artifacts not found.")

    tokens = raw_data["tokens"]

    from .section_p1 import apply_heading_predictions
    class PredProxy:
        def __init__(self, page, lineIndex):
            self.key = (page, lineIndex)
            self.is_heading = True

    proxies = [PredProxy(h["page"], h["lineIndex"]) for h in p1_data.get("headings", [])]
    apply_heading_predictions(tokens, proxies)

    from .section_p2 import run_section_phase2
    run_section_phase2(tokens)

    from .section_p1.data_utils import sort_tokens_by_reading_order
    tokens[:] = sort_tokens_by_reading_order(tokens)

    boundary_tok_map: dict[tuple, str] = {}
    for t in p1_boundaries.get("tokens", []):
        key = (t["page"], t["lineIndex"], t["tokenIndex"])
        boundary_tok_map[key] = t["prediction"]

    for t in tokens:
        key = (t.get("page"), t.get("lineIndex"), t.get("tokenIndex"))
        if key in boundary_tok_map:
            t["bioLabel"] = boundary_tok_map[key]
            t["bio_label"] = boundary_tok_map[key]

    from .overlay_mongo_labels import overlay_mongo_field_labels
    exp_indices = [
        i for i, t in enumerate(tokens)
        if t.get("section") == "EXPERIENCE"
        and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]
    overlay_mongo_field_labels([tokens[i] for i in exp_indices], raw_data.get("resumeId") or slug)

    from .experience_phase1_segment import run_experience_phase1_segment
    exp_out = run_experience_phase1_segment(tokens, raw_data.get("resumeId") or slug)
    storage.write_json(slug, EXPERIENCE_SEGMENTS_ARTIFACT, exp_out)

    return {
        "status": "completed",
        "tokenCount": exp_out["tokenCount"],
        "nonOCount": exp_out["nonOCount"],
    }


@router.post("/runs/{slug}/rerun/experience_p2")
def rerun_experience_p2(slug: str) -> dict[str, Any]:
    """Deprecated alias — use /rerun/experience_phase1_segment."""
    return rerun_experience_phase1_segment(slug)


@router.post("/runs/{slug}/rerun/experience_phase3_classify")
def rerun_experience_phase3_classify(slug: str) -> dict[str, Any]:
    """Re-run experience training phase 3 (segment field classification)."""
    try:
        raw_data = storage.read_json(slug, "1_extracted_tokens.json")
        p1_data = storage.read_json(slug, "2_section_headings.json")
        try:
            p1_boundaries = _read_artifact_legacy(
                slug, EXPERIENCE_BOUNDARIES_ARTIFACT, EXPERIENCE_BOUNDARIES_LEGACY,
            )
            p2_segments = _read_artifact_legacy(
                slug, EXPERIENCE_SEGMENTS_ARTIFACT, EXPERIENCE_SEGMENTS_LEGACY,
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"{EXPERIENCE_BOUNDARIES_ARTIFACT} or {EXPERIENCE_SEGMENTS_ARTIFACT} not found. "
                    "Run prior experience stages first."
                ),
            )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Required preceding artifacts not found.")

    tokens = raw_data["tokens"]

    from .section_p1 import apply_heading_predictions
    class PredProxy:
        def __init__(self, page, lineIndex):
            self.key = (page, lineIndex)
            self.is_heading = True

    proxies = [PredProxy(h["page"], h["lineIndex"]) for h in p1_data.get("headings", [])]
    apply_heading_predictions(tokens, proxies)

    from .section_p2 import run_section_phase2
    run_section_phase2(tokens)

    from .section_p1.data_utils import sort_tokens_by_reading_order
    tokens[:] = sort_tokens_by_reading_order(tokens)

    boundary_tok_map: dict[tuple, str] = {}
    for t in p1_boundaries.get("tokens", []):
        key = (t["page"], t["lineIndex"], t["tokenIndex"])
        boundary_tok_map[key] = t["prediction"]

    for t in tokens:
        key = (t.get("page"), t.get("lineIndex"), t.get("tokenIndex"))
        if key in boundary_tok_map:
            t["bioLabel"] = boundary_tok_map[key]
            t["bio_label"] = boundary_tok_map[key]

    segment_tok_map: dict[tuple, str] = {}
    for t in p2_segments.get("tokens", []):
        key = (t["page"], t["lineIndex"], t["tokenIndex"])
        segment_tok_map[key] = t["prediction"]

    for t in tokens:
        key = (t.get("page"), t.get("lineIndex"), t.get("tokenIndex"))
        if key in segment_tok_map:
            t["segLabel"] = segment_tok_map[key]

    from .experience_phase3_classify import run_experience_phase3_classify
    exp_out = run_experience_phase3_classify(tokens, raw_data.get("resumeId") or slug, slug=slug)
    storage.write_json(slug, EXPERIENCE_CLASSIFY_ARTIFACT, exp_out)

    return {
        "status": "completed",
        "tokenCount": exp_out["tokenCount"],
        "nonOCount": exp_out["nonOCount"],
    }


@router.post("/runs/{slug}/rerun/experience_p3")
def rerun_experience_p3(slug: str) -> dict[str, Any]:
    """Deprecated alias — use /rerun/experience_phase3_classify."""
    return rerun_experience_phase3_classify(slug)


@router.post("/experience_phase1_segment/predict")
def predict_experience_phase1_segment(req: TokensRequest) -> dict[str, Any]:
    from .experience_phase1_segment import run_experience_phase1_segment
    tokens = [dict(t) for t in req.tokens]
    return run_experience_phase1_segment(tokens)


@router.post("/experience_p2/predict")
def predict_experience_p2(req: TokensRequest) -> dict[str, Any]:
    """Deprecated alias — use /experience_phase1_segment/predict."""
    return predict_experience_phase1_segment(req)


@router.post("/experience_phase3_classify/predict")
def predict_experience_phase3_classify(req: TokensRequest) -> dict[str, Any]:
    from .experience_phase3_classify import run_experience_phase3_classify
    tokens = [dict(t) for t in req.tokens]
    return run_experience_phase3_classify(tokens)


@router.post("/experience_p3/predict")
def predict_experience_p3(req: TokensRequest) -> dict[str, Any]:
    """Deprecated alias — use /experience_phase3_classify/predict."""
    return predict_experience_phase3_classify(req)


@router.post("/runs/{slug}/rerun/project_phase2_divider")
def rerun_project_phase2_divider(slug: str) -> dict[str, Any]:
    try:
        raw_data = storage.read_json(slug, "1_extracted_tokens.json")
        p1_data = storage.read_json(slug, "2_section_headings.json")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Required preceding artifacts not found.")

    tokens = raw_data["tokens"]

    from .section_p1 import apply_heading_predictions
    class PredProxy:
        def __init__(self, page, lineIndex):
            self.key = (page, lineIndex)
            self.is_heading = True

    proxies = [PredProxy(h["page"], h["lineIndex"]) for h in p1_data.get("headings", [])]
    apply_heading_predictions(tokens, proxies)

    from .section_p2 import run_section_phase2
    run_section_phase2(tokens)

    from .section_p1.data_utils import sort_tokens_by_reading_order
    tokens[:] = sort_tokens_by_reading_order(tokens)

    from .project_phase2_divider import run_project_phase2_divider
    resume_id = raw_data.get("resumeId") or slug
    proj_out = run_project_phase2_divider(tokens, resume_id)
    storage.write_json(slug, "12_project_boundaries.json", proj_out)

    return {
        "status": "completed",
        "tokenCount": proj_out["tokenCount"],
        "nonOCount": proj_out["nonOCount"],
    }


@router.post("/runs/{slug}/rerun/project_p1")
def rerun_project_p1(slug: str) -> dict[str, Any]:
    """Deprecated alias — use /rerun/project_phase2_divider."""
    return rerun_project_phase2_divider(slug)


@router.post("/project_phase2_divider/predict")
def predict_project_phase2_divider(req: TokensRequest) -> dict[str, Any]:
    from .project_phase2_divider import run_project_phase2_divider
    tokens = [dict(t) for t in req.tokens]
    return run_project_phase2_divider(tokens)


@router.post("/project_p1/predict")
def predict_project_p1(req: TokensRequest) -> dict[str, Any]:
    """Deprecated alias — use /project_phase2_divider/predict."""
    return predict_project_phase2_divider(req)


@router.post("/runs/{slug}/rerun/project_phase1_segment")
def rerun_project_phase1_segment(slug: str) -> dict[str, Any]:
    """Re-run project training phase 1 (token segmentation)."""
    try:
        raw_data = storage.read_json(slug, "1_extracted_tokens.json")
        p1_data = storage.read_json(slug, "2_section_headings.json")
        try:
            boundary_data = storage.read_json(slug, "12_project_boundaries.json")
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="12_project_boundaries.json not found. Run project_phase2_divider first.",
            )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Required preceding artifacts not found.")

    tokens = raw_data["tokens"]

    from .section_p1 import apply_heading_predictions
    class PredProxy:
        def __init__(self, page, lineIndex):
            self.key = (page, lineIndex)
            self.is_heading = True

    proxies = [PredProxy(h["page"], h["lineIndex"]) for h in p1_data.get("headings", [])]
    apply_heading_predictions(tokens, proxies)

    from .section_p2 import run_section_phase2
    run_section_phase2(tokens)

    from .section_p1.data_utils import sort_tokens_by_reading_order
    tokens[:] = sort_tokens_by_reading_order(tokens)

    boundary_tok_map: dict[tuple, str] = {}
    for t in boundary_data.get("tokens", []):
        key = (t["page"], t["lineIndex"], t["tokenIndex"])
        boundary_tok_map[key] = t.get("prediction", "O")

    for t in tokens:
        key = (t.get("page"), t.get("lineIndex"), t.get("tokenIndex"))
        if key in boundary_tok_map:
            t["bioLabel"] = boundary_tok_map[key]
            t["bio_label"] = boundary_tok_map[key]

    from .overlay_mongo_labels import overlay_mongo_field_labels
    proj_indices = [
        i for i, t in enumerate(tokens)
        if t.get("section") in ("PROJECT", "PROJECTS")
        and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]
    overlay_mongo_field_labels([tokens[i] for i in proj_indices], raw_data.get("resumeId") or slug)

    from .project_phase1_segment import run_project_phase1_segment
    proj_out = run_project_phase1_segment(tokens, raw_data.get("resumeId") or slug)
    storage.write_json(slug, "11_project_segments.json", proj_out)

    return {
        "status": "completed",
        "tokenCount": proj_out["tokenCount"],
        "nonOCount": proj_out["nonOCount"],
    }


@router.post("/runs/{slug}/rerun/project_p2")
def rerun_project_p2(slug: str) -> dict[str, Any]:
    """Deprecated alias — use /rerun/project_phase1_segment."""
    return rerun_project_phase1_segment(slug)


@router.post("/project_phase1_segment/predict")
def predict_project_phase1_segment(req: TokensRequest) -> dict[str, Any]:
    from .project_phase1_segment import run_project_phase1_segment
    tokens = [dict(t) for t in req.tokens]
    return run_project_phase1_segment(tokens)


@router.post("/project_p2/predict")
def predict_project_p2(req: TokensRequest) -> dict[str, Any]:
    """Deprecated alias — use /project_phase1_segment/predict."""
    return predict_project_phase1_segment(req)


@router.post("/runs/{slug}/rerun/project_phase3_classify")
def rerun_project_phase3_classify(slug: str) -> dict[str, Any]:
    """Re-run project training phase 3 (segment field classification)."""
    try:
        raw_data = storage.read_json(slug, "1_extracted_tokens.json")
        p1_data = storage.read_json(slug, "2_section_headings.json")
        boundary_data = storage.read_json(slug, "12_project_boundaries.json")
        segment_data = storage.read_json(slug, "11_project_segments.json")
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Required preceding artifacts not found: {exc}",
        )

    tokens = raw_data["tokens"]

    from .section_p1 import apply_heading_predictions

    class PredProxy:
        def __init__(self, page, lineIndex):
            self.key = (page, lineIndex)
            self.is_heading = True

    proxies = [PredProxy(h["page"], h["lineIndex"]) for h in p1_data.get("headings", [])]
    apply_heading_predictions(tokens, proxies)

    from .section_p2 import run_section_phase2
    run_section_phase2(tokens)

    from .section_p1.data_utils import sort_tokens_by_reading_order
    tokens[:] = sort_tokens_by_reading_order(tokens)

    boundary_tok_map: dict[tuple, str] = {}
    for t in boundary_data.get("tokens", []):
        key = (t["page"], t["lineIndex"], t["tokenIndex"])
        boundary_tok_map[key] = t.get("prediction", "O")

    for t in tokens:
        key = (t.get("page"), t.get("lineIndex"), t.get("tokenIndex"))
        if key in boundary_tok_map:
            t["bioLabel"] = boundary_tok_map[key]
            t["bio_label"] = boundary_tok_map[key]

    segment_tok_map: dict[tuple, str] = {}
    for t in segment_data.get("tokens", []):
        key = (t["page"], t["lineIndex"], t["tokenIndex"])
        segment_tok_map[key] = t.get("prediction", "O")

    for t in tokens:
        key = (t.get("page"), t.get("lineIndex"), t.get("tokenIndex"))
        if key in segment_tok_map:
            t["segLabel"] = segment_tok_map[key]

    resume_id = raw_data.get("resumeId") or slug
    from .project_phase3_classify import run_project_phase3_classify
    proj_out = run_project_phase3_classify(tokens, resume_id)
    storage.write_json(slug, "13_project_fields.json", proj_out)

    from .entities import build_entities_dict
    structured = build_entities_dict(tokens, resume_id)
    if p1_data.get("headings"):
        structured["SECTION_HEADINGS"] = [h["text"] for h in p1_data["headings"]]
    storage.write_json(slug, "14_final_classified_tokens.json", {"resumeId": resume_id, "tokens": tokens})
    storage.write_json(slug, "structured.json", {"resumeId": resume_id, "entities": structured})

    return {
        "status": "completed",
        "tokenCount": proj_out["tokenCount"],
        "nonOCount": proj_out["nonOCount"],
        "segmentAccuracy": proj_out.get("blockClassification", {}).get("metrics", {}).get("segmentAccuracyPercent"),
    }


@router.post("/runs/{slug}/rerun/project_p3")
def rerun_project_p3(slug: str) -> dict[str, Any]:
    """Deprecated alias — use /rerun/project_phase3_classify."""
    return rerun_project_phase3_classify(slug)


@router.post("/project_phase3_classify/predict")
def predict_project_phase3_classify(req: TokensRequest) -> dict[str, Any]:
    from .project_phase3_classify import run_project_phase3_classify
    tokens = [dict(t) for t in req.tokens]
    return run_project_phase3_classify(tokens)


@router.post("/project_p3/predict")
def predict_project_p3(req: TokensRequest) -> dict[str, Any]:
    """Deprecated alias — use /project_phase3_classify/predict."""
    return predict_project_phase3_classify(req)

