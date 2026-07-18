"""Segment-level classification report aligned to step 4 B-SEG blocks."""

from __future__ import annotations

import os

from .config import FINAL_LABEL_LIST
from .data_utils import block_to_segment, coord_key
from .training_bridge import load_training_helpers
from ..gt_gate import is_gt_enabled


def _mongo_tokens_by_coord(mongo_tokens: list[dict]) -> dict[tuple[int, float, float], dict]:
    return {coord_key(token): token for token in mongo_tokens}


def _gt_label_for_block(
    block_tokens: list[dict],
    mongo_by_coord: dict[tuple[int, float, float], dict],
    helpers: dict,
) -> str | None:
    matched = [mongo_by_coord[key] for token in block_tokens if (key := coord_key(token)) in mongo_by_coord]
    if not matched:
        return None
    segment = block_to_segment(
        matched,
        helpers["clean_cid_tokens"],
        helpers["construct_sentences_by_appearance"],
    )
    if not segment:
        return None
    return helpers["get_segment_class_label"](segment)


def build_segment_classification_report(
    resume_id: str,
    education_segments: list[dict],
    pred_labels: list[str],
    pred_confidences: list[float] | None = None,
) -> dict:
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGO_DB", "resume-labeling")
    helpers = load_training_helpers()

    mongo_by_coord: dict[tuple[int, float, float], dict] = {}
    if is_gt_enabled(resume_id):
        try:
            from pymongo import MongoClient

            from ..overlay_mongo_labels import resolve_mongo_resume_id

            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            doc = client[mongo_db]["resumes"].find_one({"resumeId": resolve_mongo_resume_id(resume_id)})
            client.close()
            if doc:
                mongo_edu = [
                    token for token in doc.get("tokens", [])
                    if token.get("section") == "EDUCATION"
                    and token.get("bioLabel") not in ("B-HEADING", "I-HEADING")
                ]
                mongo_by_coord = _mongo_tokens_by_coord(mongo_edu)
        except Exception:
            pass

    scored: list[dict] = []
    correct = 0
    total = min(len(education_segments), len(pred_labels))

    for index in range(total):
        segment = education_segments[index]
        block_tokens = segment.get("tokens") or []
        text = (segment.get("text") or "").strip()
        pred = pred_labels[index]
        gt = _gt_label_for_block(block_tokens, mongo_by_coord, helpers) if mongo_by_coord else None
        if gt is None:
            continue
        match = gt == pred
        if match:
            correct += 1
        conf = (
            pred_confidences[index]
            if pred_confidences is not None and index < len(pred_confidences)
            else 0.0
        )
        scored.append({
            "status": "✅" if match else "❌",
            "entryKey": f"Seg {index}",
            "gt": gt,
            "pred": pred,
            "confidence": conf,
            "text": text[:140],
        })

    seg_total = len(scored)
    accuracy = (correct / seg_total * 100.0) if seg_total else 0.0

    return {
        "gtSource": "mongodb.tokens.bioLabel (coordinate-matched B-SEG block majority)",
        "trainingReport": "education/new_phase3_segment_classification/reports/minilm/per_resume/*.md",
        "macroClasses": FINAL_LABEL_LIST,
        "metrics": {
            "macroF1ProxyPercent": round(accuracy, 2),
            "segmentAccuracyPercent": round(accuracy, 2),
            "blocks": seg_total,
            "segments": seg_total,
            "correct": correct,
            "errors": seg_total - correct,
        },
        "blockRows": scored,
        "gtSegmentCount": seg_total,
        "predSegmentCount": total,
    }
