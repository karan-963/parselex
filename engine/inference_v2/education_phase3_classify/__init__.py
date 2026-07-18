"""Education training phase 3 — segment field classification (step 6).

Groups EDUCATION tokens into step 4 B-SEG phrase blocks within step 5 entry
boundaries, classifies each block with PhraseSegmentClassifierModel, applies
block heuristics, then stamps B-INST / B-DEG / B-SDATE / O on tokens.
"""

from __future__ import annotations

from typing import Any

from inference_v2.education_phase2_divider.entry_slice_heads import resolve_education_boundary_heads
from inference_v2.predictor_cache import get_predictor

from .bio_stamps import apply_bio_stamps
from .config import FINAL_LABEL_LIST
from .data_utils import build_education_block_segments
from .predictor import EducationPhase3Predictor
from .segment_classification_report import build_segment_classification_report


def _is_education_token(token: dict) -> bool:
    return token.get("section") == "EDUCATION"


def _empty_result(tokens: list[dict], filtered_indices: list[int], resume_id: str, reason: str = "empty") -> dict[str, Any]:
    return {
        "stage": "education_phase3_classify",
        "title": "Education Field Classification",
        "section": "EDUCATION",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "education/new_phase3_segment_classification",
        "task": "field_classification",
        "labelField": "prediction",
        "labels": FINAL_LABEL_LIST,
        "reason": reason,
        "tokenCount": len(filtered_indices),
        "evalTokenCount": len(filtered_indices),
        "nonOCount": 0,
        "sampleLabels": [],
        "tokens": _serialize_tokens(tokens, filtered_indices),
    }


def _serialize_tokens(tokens: list[dict], indices: list[int]) -> list[dict]:
    return [
        {
            "page": tokens[idx]["page"],
            "lineIndex": tokens[idx]["lineIndex"],
            "tokenIndex": tokens[idx]["tokenIndex"],
            "token": tokens[idx]["token"],
            "prediction": tokens[idx].get("bioLabel", "O"),
            "confidence": tokens[idx].get("confidence", 0.0),
            "x0": tokens[idx].get("x0"),
            "y0": tokens[idx].get("y0"),
            "x1": tokens[idx].get("x1"),
            "y1": tokens[idx].get("y1"),
        }
        for idx in indices
    ]


def run_education_phase3_classify(tokens: list[dict], resume_id: str = "resume") -> dict[str, Any]:
    filtered_indices = [
        i for i, t in enumerate(tokens)
        if _is_education_token(t)
        and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]

    if len(filtered_indices) < 2:
        return _empty_result(tokens, filtered_indices, resume_id, reason="insufficient_education_tokens")

    filtered_tokens = [tokens[idx] for idx in filtered_indices]
    edu_keys = {(t.get("page"), t.get("lineIndex")) for t in filtered_tokens}
    head_lines = resolve_education_boundary_heads(filtered_tokens) & edu_keys
    if not head_lines and any(t.get("segLabel") == "B-SEG" for t in filtered_tokens):
        return _empty_result(tokens, filtered_indices, resume_id, reason="no_education_entry_boundaries")

    predictor = get_predictor("education_phase3_classify", EducationPhase3Predictor)
    helpers = predictor._helpers
    education_segments = build_education_block_segments(
        filtered_tokens,
        head_lines,
        helpers["clean_cid_tokens"],
        helpers["construct_sentences_by_appearance"],
        helpers["is_education_segment"],
    )
    if not education_segments:
        return _empty_result(tokens, filtered_indices, resume_id, reason="no_education_segments")

    resolved_labels, education_segments, pred_confs = predictor.classify_education_blocks(education_segments)

    if not education_segments:
        return _empty_result(tokens, filtered_indices, resume_id, reason="no_education_segments")

    for segment, conf in zip(education_segments, pred_confs):
        for tok in segment.get("tokens", []):
            if tok:
                tok["confidence"] = conf

    final_labels = apply_bio_stamps(education_segments, resolved_labels)

    for idx in filtered_indices:
        t = tokens[idx]
        pred_bio = final_labels.get(id(t), "O")
        t["bioLabel"] = pred_bio
        t["bio_label"] = pred_bio

    segment_report = build_segment_classification_report(
        resume_id, education_segments, resolved_labels, pred_confs,
    )
    non_o_count = sum(1 for idx in filtered_indices if tokens[idx]["bioLabel"] != "O")

    return {
        "stage": "education_phase3_classify",
        "title": "Education Field Classification",
        "section": "EDUCATION",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "education/new_phase3_segment_classification",
        "task": "field_classification",
        "labelField": "prediction",
        "labels": FINAL_LABEL_LIST,
        "blockClassification": segment_report,
        "tokenCount": len(filtered_indices),
        "evalTokenCount": len(filtered_indices),
        "nonOCount": non_o_count,
        "sampleLabels": sorted({tokens[idx]["bioLabel"] for idx in filtered_indices}),
        "tokens": _serialize_tokens(tokens, filtered_indices),
    }
