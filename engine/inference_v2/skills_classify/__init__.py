"""Skills direct token BIO classification (step 7).

Runs SkillsSegmentClassifierModel on SKILLS-section tokens, stamps
B-SKILL / I-SKILL / B-SKILL_TYPE / I-SKILL_TYPE / O on each token.
"""

from __future__ import annotations

from typing import Any

from inference_v2.predictor_cache import get_predictor

from .config import LABEL_LIST
from .predictor import SkillsPhase7Predictor
from .token_classification_report import build_token_classification_report


def _is_skills_token(token: dict) -> bool:
    return token.get("section") == "SKILLS"


def _empty_result(tokens: list[dict], filtered_indices: list[int], resume_id: str, reason: str = "empty") -> dict[str, Any]:
    return {
        "stage": "skills_classify",
        "title": "Skills Token Classification",
        "section": "SKILLS",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "skills/",
        "task": "token_classification",
        "labelField": "prediction",
        "labels": LABEL_LIST,
        "reason": reason,
        "tokenCount": len(filtered_indices),
        "evalTokenCount": 0,
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


def run_skills_classify(tokens: list[dict], resume_id: str = "resume") -> dict[str, Any]:
    filtered_indices = [
        index for index, token in enumerate(tokens)
        if _is_skills_token(token)
        and token.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]

    if len(filtered_indices) < 3:
        return _empty_result(tokens, filtered_indices, resume_id, reason="insufficient_skills_tokens")

    predictor = get_predictor("skills_classify", SkillsPhase7Predictor)
    all_pred_labels, token_confs = predictor.classify_tokens(tokens, all_tokens=tokens)

    for index in filtered_indices:
        token = tokens[index]
        pred_bio = all_pred_labels[index]
        token["bioLabel"] = pred_bio
        token["bio_label"] = pred_bio
        token["confidence"] = token_confs.get(id(token), 0.0)

    skills_tokens = [tokens[index] for index in filtered_indices]
    pred_labels = [tokens[index]["bioLabel"] for index in filtered_indices]

    report_tokens = skills_tokens

    token_report = build_token_classification_report(resume_id, report_tokens, pred_labels)
    non_o_count = sum(1 for index in filtered_indices if tokens[index]["bioLabel"] != "O")

    return {
        "stage": "skills_classify",
        "title": "Skills Token Classification",
        "section": "SKILLS",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "skills/",
        "task": "token_classification",
        "labelField": "prediction",
        "labels": LABEL_LIST,
        "tokenClassification": token_report,
        "tokenCount": len(filtered_indices),
        "evalTokenCount": token_report["metrics"]["evalTokens"],
        "nonOCount": non_o_count,
        "sampleLabels": sorted({tokens[index]["bioLabel"] for index in filtered_indices}),
        "tokens": _serialize_tokens(tokens, filtered_indices),
    }
