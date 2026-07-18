"""Project training phase 3 — segment field classification (step 13).

Groups PROJECT tokens into phrase segments (construct_sentences_by_appearance),
classifies each segment with PhraseSegmentClassifierModel, applies block heuristics
and DATE → SDATE/EDATE resolution, then stamps B-PROJ_NAME / B-DESC / etc. on tokens.
"""

from __future__ import annotations

from typing import Any

from inference_v2.predictor_cache import get_predictor

from .bio_stamps import apply_bio_stamps
from .config import FINAL_LABEL_LIST
from .predictor import ProjectPhase3Predictor
from .segment_classification_report import build_segment_classification_report


def _is_project_token(token: dict) -> bool:
    return token.get("section") in ("PROJECT", "PROJECTS")


def _empty_result(tokens: list[dict], filtered_indices: list[int], resume_id: str, reason: str = "empty") -> dict[str, Any]:
    return {
        "stage": "project_phase3_classify",
        "title": "Project Field Classification",
        "section": "PROJECT",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "project/phase3_segment_classification",
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


def run_project_phase3_classify(tokens: list[dict], resume_id: str = "resume") -> dict[str, Any]:
    filtered_indices = [
        i for i, t in enumerate(tokens)
        if _is_project_token(t)
        and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]

    if len(filtered_indices) < 2:
        return _empty_result(tokens, filtered_indices, resume_id, reason="insufficient_project_tokens")

    filtered_tokens = [tokens[idx] for idx in filtered_indices]

    # Preserve step-12 entry heads before field labels overwrite boundary bioLabels.
    from inference_v2.project_phase2_divider.entry_slice_heads import resolve_project_entry_heads

    head_lines = {
        (int(t.get("page", 0)), int(t.get("lineIndex", t.get("line_index", 0))))
        for t in filtered_tokens
        if t.get("_projEntryHead")
    }
    if not head_lines:
        head_lines = resolve_project_entry_heads(filtered_tokens)
    for t in filtered_tokens:
        key = (int(t.get("page", 0)), int(t.get("lineIndex", t.get("line_index", 0))))
        if key in head_lines:
            t["_projEntryHead"] = True

    predictor = get_predictor("project_phase3_classify", ProjectPhase3Predictor)
    _, resolved_labels, project_segments, pred_confs = predictor.classify_segments(tokens)

    if not project_segments:
        return _empty_result(tokens, filtered_indices, resume_id, reason="no_project_segments")

    for segment, conf in zip(project_segments, pred_confs):
        for tok in segment.get("tokens", []):
            if tok:
                tok["confidence"] = conf

    final_labels = apply_bio_stamps(project_segments, resolved_labels)

    for idx in filtered_indices:
        t = tokens[idx]
        pred_bio = final_labels.get(id(t), "O")
        t["bioLabel"] = pred_bio
        t["bio_label"] = pred_bio

    segment_report = build_segment_classification_report(
        resume_id, project_segments, resolved_labels, pred_confs,
    )

    non_o_count = sum(1 for idx in filtered_indices if tokens[idx]["bioLabel"] != "O")

    return {
        "stage": "project_phase3_classify",
        "title": "Project Field Classification",
        "section": "PROJECT",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "project/phase3_segment_classification",
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
