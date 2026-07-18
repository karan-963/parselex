"""Personal training segment field classification (step 15).

Builds atomic PERSONAL segments via heuristic segmentation, classifies each
segment with PersonalSegmentClassifierModel, applies text heuristics, then
stamps B-NAME / B-EMAIL / … onto tokens.
"""

from __future__ import annotations

from typing import Any

from inference_v2.predictor_cache import get_predictor

from .bio_stamps import apply_bio_stamps
from .config import FINAL_LABEL_LIST
from .predictor import PersonalPhase15Predictor
from .segment_classification_report import build_segment_classification_report
from .token_regex_refine import refine_personal_token_labels
from .training_bridge import load_training_helpers


def _is_personal_token(token: dict) -> bool:
    return token.get("section") == "PERSONAL"


def _empty_result(tokens: list[dict], filtered_indices: list[int], resume_id: str, reason: str = "empty") -> dict[str, Any]:
    return {
        "stage": "personal_classify",
        "title": "Personal Field Classification",
        "section": "PERSONAL",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "personal",
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


def run_personal_classify(tokens: list[dict], resume_id: str = "resume") -> dict[str, Any]:
    filtered_indices = [
        index for index, token in enumerate(tokens)
        if _is_personal_token(token)
        and token.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]

    if not filtered_indices:
        return _empty_result(tokens, filtered_indices, resume_id, reason="no_personal_tokens")

    helpers = load_training_helpers()
    cleaned = helpers["clean_cid_tokens"](tokens)
    cleaned.sort(key=lambda token: (token.get("page", 0), token.get("y0", 0.0), token.get("x0", 0.0)))
    personal_segments = helpers["build_personal_segments"](cleaned)
    if not personal_segments:
        return _empty_result(tokens, filtered_indices, resume_id, reason="no_personal_segments")

    predictor = get_predictor("personal_classify", PersonalPhase15Predictor)
    resolved_labels, personal_segments, pred_confs = predictor.classify_personal_segments(
        personal_segments, cleaned,
    )
    if not personal_segments:
        return _empty_result(tokens, filtered_indices, resume_id, reason="no_personal_segments")

    for segment, conf in zip(personal_segments, pred_confs):
        for tok in segment.get("tokens", []):
            if tok:
                tok["confidence"] = conf

    final_labels = apply_bio_stamps(personal_segments, resolved_labels)
    for index in filtered_indices:
        token = tokens[index]
        pred_bio = final_labels.get(id(token), "O")
        token["bioLabel"] = pred_bio
        token["bio_label"] = pred_bio

    # Split mixed segments at the token level: re-label unambiguous contact atoms
    # (phone / email / URL) by their own text so a phone glued to an email no
    # longer inherits the segment's EMAIL label.
    refine_personal_token_labels(tokens, filtered_indices)

    segment_report = build_segment_classification_report(
        resume_id, personal_segments, resolved_labels, pred_confs,
    )
    non_o_count = sum(1 for index in filtered_indices if tokens[index]["bioLabel"] != "O")

    return {
        "stage": "personal_classify",
        "title": "Personal Field Classification",
        "section": "PERSONAL",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "personal",
        "task": "field_classification",
        "labelField": "prediction",
        "labels": FINAL_LABEL_LIST,
        "blockClassification": segment_report,
        "tokenCount": len(filtered_indices),
        "evalTokenCount": len(filtered_indices),
        "nonOCount": non_o_count,
        "sampleLabels": sorted({tokens[index]["bioLabel"] for index in filtered_indices}),
        "tokens": _serialize_tokens(tokens, filtered_indices),
    }
