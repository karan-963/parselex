"""Segmentation metrics for education phase 1 artifact."""

from __future__ import annotations

import re
from typing import Any

STRUCTURAL_TOKENS = frozenset({"|", "•", "-", "–", "—", "*", "▪", "◦", "■", "·", ",", "✓", "✔", '"'})


def derive_seg_gt(bio: str | None) -> str:
    if not bio or bio == "O" or "HEADING" in bio:
        return "O"
    return "B-SEG" if bio.startswith("B-") else "I-SEG"


def is_eval_token(token_str: str) -> bool:
    tok = (token_str or "").strip()
    if tok in STRUCTURAL_TOKENS:
        return False
    return bool(re.search(r"[a-zA-Z0-9]", tok))


def build_token_segmentation_metrics(filtered_tokens: list[dict]) -> dict[str, Any]:
    correct = 0
    eval_total = 0
    for t in filtered_tokens:
        if not is_eval_token(t.get("token", "")):
            continue
        bio = t.get("_fieldBioLabel") or t.get("bioLabel") or "O"
        gt = derive_seg_gt(bio)
        pred = t.get("prediction", "O")
        eval_total += 1
        if gt == pred:
            correct += 1
    accuracy = (correct / eval_total * 100.0) if eval_total else 0.0
    return {
        "gtSource": "mongodb.bioLabel→B-SEG/I-SEG",
        "trainingReport": "education/new_phase1_token_segmentation/reports/minilm/per_resume/*.md",
        "metrics": {
            "tokenAccuracyPercent": round(accuracy, 2),
            "correct": correct,
            "evalTokens": eval_total,
            "scoringNote": "alphanumeric tokens only (matches training eval)",
        },
    }
