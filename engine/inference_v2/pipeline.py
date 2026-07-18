"""Simplified PDF inference pipeline orchestrator for code isolation validation.

Runs implemented stages only; pending stages are documented in config.PIPELINE_STEPS."""

from __future__ import annotations

import traceback
from collections import Counter
from typing import Any

from . import storage
from .entities import build_entities_dict
from .extract import extract_tokens_from_pdf
from .model_precision import set_precision
from .performance import PipelinePerformanceTracker
from .performance_metrics import enrich_performance_summary
from .section_p1 import run_section_phase1
from .section_p2 import run_section_phase2
from .education_phase2_divider import run_education_phase2_divider
from .education_phase1_segment import run_education_phase1_segment
from .education_phase3_classify import run_education_phase3_classify
from .skills_classify import run_skills_classify
from .experience_phase2_divider import run_experience_phase2_divider
from .experience_phase1_segment import run_experience_phase1_segment
from .experience_phase3_classify import run_experience_phase3_classify
from .project_phase2_divider import run_project_phase2_divider
from .project_phase1_segment import run_project_phase1_segment
from .project_phase3_classify import run_project_phase3_classify
from .personal_classify import run_personal_classify


def run_pipeline(slug: str, pdf_path: str, precision: str = "fp32") -> dict[str, Any]:
    set_precision(precision)
    storage.copy_input_pdf(slug, pdf_path)
    manifest = storage.read_manifest(slug) or {}
    original_name = manifest.get("originalFilename") or pdf_path
    resume_id = storage.sanitize_basename(original_name)
    perf = PipelinePerformanceTracker()

    try:
        with perf.stage("extract_tokens"):
            storage.update_stage(slug, "extract_tokens")
            tokens, doc_id = extract_tokens_from_pdf(pdf_path)
            storage.write_json(
                slug,
                "1_extracted_tokens.json",
                {"resumeId": resume_id, "docId": doc_id, "tokens": tokens},
            )

        orig_by_coord: dict[tuple[Any, float, float], Any] = {}
        for t in tokens:
            key = (t.get("page"), round(float(t.get("x0", 0)), 2), round(float(t.get("y0", 0)), 2))
            orig_by_coord[key] = t.get("lineIndex")

        with perf.stage("section_phase1"):
            storage.update_stage(slug, "section_phase1")
            p1_out = run_section_phase1(tokens)
            storage.write_json(slug, "2_section_headings.json", p1_out)

        with perf.stage("section_phase2"):
            storage.update_stage(slug, "section_phase2")
            p2_out = run_section_phase2(tokens)
            section_counts = Counter(t.get("section") for t in tokens)
            p2_out["sectionTokenCounts"] = dict(section_counts)
            if len(section_counts) == 1 and section_counts.get("PERSONAL", 0) == len(tokens) and len(p2_out.get("chunks", [])) > 1:
                raise RuntimeError(
                    "Section phase 2 classified chunks but failed to assign section labels to tokens. "
                    "Check section_p2 token_map propagation."
                )
            storage.write_json(slug, "3_section_labels.json", p2_out)

        for t in tokens:
            key = (t.get("page"), round(float(t.get("x0", 0)), 2), round(float(t.get("y0", 0)), 2))
            if key in orig_by_coord:
                t["lineIndex"] = orig_by_coord[key]

        from .section_p1.data_utils import sort_tokens_by_reading_order
        tokens[:] = sort_tokens_by_reading_order(tokens)

        with perf.stage("education_phase2_divider"):
            storage.update_stage(slug, "education_phase2_divider")
            edu_divider_out = run_education_phase2_divider(tokens, resume_id)
            storage.write_json(slug, "5_education_boundaries.json", edu_divider_out)

        with perf.stage("education_phase1_segment"):
            storage.update_stage(slug, "education_phase1_segment")
            edu_segment_out = run_education_phase1_segment(tokens, resume_id)
            storage.write_json(slug, "4_education_segments.json", edu_segment_out)

        with perf.stage("education_phase3_classify"):
            storage.update_stage(slug, "education_phase3_classify")
            edu_classify_out = run_education_phase3_classify(tokens, resume_id)
            storage.write_json(slug, "6_education_fields.json", edu_classify_out)

        with perf.stage("skills_classify"):
            storage.update_stage(slug, "skills_classify")
            skills_out = run_skills_classify(tokens, resume_id)
            storage.write_json(slug, "7_skills_fields.json", skills_out)

        with perf.stage("experience_phase2_divider"):
            storage.update_stage(slug, "experience_phase2_divider")
            exp_divider_out = run_experience_phase2_divider(tokens, resume_id)
            storage.write_json(slug, "9_experience_boundaries.json", exp_divider_out)

        with perf.stage("experience_phase1_segment"):
            storage.update_stage(slug, "experience_phase1_segment")
            exp_segment_out = run_experience_phase1_segment(tokens, resume_id, slug=slug)
            storage.write_json(slug, "8_experience_segments.json", exp_segment_out)

        with perf.stage("experience_phase3_classify"):
            storage.update_stage(slug, "experience_phase3_classify")
            exp_classify_out = run_experience_phase3_classify(tokens, resume_id, slug=slug)
            storage.write_json(slug, "10_experience_classification.json", exp_classify_out)

        with perf.stage("project_phase2_divider"):
            storage.update_stage(slug, "project_phase2_divider")
            proj_divider_out = run_project_phase2_divider(tokens, resume_id)
            storage.write_json(slug, "12_project_boundaries.json", proj_divider_out)

        with perf.stage("project_phase1_segment"):
            storage.update_stage(slug, "project_phase1_segment")
            proj_segment_out = run_project_phase1_segment(tokens, resume_id)
            storage.write_json(slug, "11_project_segments.json", proj_segment_out)

        with perf.stage("project_phase3_classify"):
            storage.update_stage(slug, "project_phase3_classify")
            proj_classify_out = run_project_phase3_classify(tokens, resume_id)
            storage.write_json(slug, "13_project_fields.json", proj_classify_out)

        with perf.stage("personal_classify"):
            storage.update_stage(slug, "personal_classify")
            personal_out = run_personal_classify(tokens, resume_id)
            storage.write_json(slug, "15_personal_fields.json", personal_out)

        with perf.stage("finalize"):
            storage.update_stage(slug, "finalize")
            structured = build_entities_dict(tokens, resume_id)
            if p1_out.get("headings"):
                structured["SECTION_HEADINGS"] = [h["text"] for h in p1_out["headings"]]

            storage.write_json(slug, "14_final_classified_tokens.json", {"resumeId": resume_id, "tokens": tokens})
            storage.write_json(slug, "structured.json", {"resumeId": resume_id, "entities": structured})

        performance_stats = perf.summary()
        performance_stats = enrich_performance_summary(
            performance_stats,
            tokens=tokens,
            stage_outputs={
                "section_phase1": p1_out,
                "section_phase2": p2_out,
                "education_phase2_divider": edu_divider_out,
                "education_phase1_segment": edu_segment_out,
                "education_phase3_classify": edu_classify_out,
                "skills_classify": skills_out,
                "experience_phase2_divider": exp_divider_out,
                "experience_phase1_segment": exp_segment_out,
                "experience_phase3_classify": exp_classify_out,
                "project_phase2_divider": proj_divider_out,
                "project_phase1_segment": proj_segment_out,
                "project_phase3_classify": proj_classify_out,
                "personal_classify": personal_out,
            },
        )
        storage.write_json(slug, "performance.json", performance_stats)

        manifest = storage.read_manifest(slug) or {}
        artifacts = manifest.get("artifacts", [])
        storage.mark_completed(slug, artifacts, performance_stats=performance_stats)

        return {
            "slug": slug,
            "status": "completed",
            "resumeId": resume_id,
            "structured": structured,
            "artifacts": artifacts,
            "performanceStats": performance_stats,
        }
    except Exception as exc:
        tb = traceback.format_exc()
        stage = (storage.read_manifest(slug) or {}).get("currentStage", "unknown")
        storage.mark_failed(slug, stage, f"{exc}\n\n{tb}")
        raise
