"""Segment-level classification report (mirrors phase3 per_resume/*.md)."""

from __future__ import annotations

import os

from .config import FINAL_LABEL_LIST
from .date_resolve import resolve_dates_to_sdate_edate
from .training_bridge import load_training_helpers
from ..gt_gate import is_gt_enabled


def _build_gt_labels(mongo_tokens: list[dict]) -> tuple[list[str], list[dict]]:
    h = load_training_helpers()
    cleaned = h["clean_cid_tokens"](mongo_tokens)
    segments = h["construct_sentences_by_appearance"](cleaned)
    segments = h["split_hyphenated_segments"](segments)

    project_segments: list[dict] = []
    raw_gt: list[str] = []
    for s in segments:
        if h["is_project_segment"](s):
            project_segments.append(s)
            raw_gt.append(h["get_segment_class_label"](s))

    gt_labels = resolve_dates_to_sdate_edate(raw_gt, project_segments)
    return gt_labels, project_segments


def build_segment_classification_report(
    resume_id: str,
    project_segments: list[dict],
    pred_labels: list[str],
    pred_confidences: list[float] | None = None,
) -> dict:
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGO_DB", "resume-labeling")

    gt_labels: list[str] = []
    gt_segments: list[dict] = []
    if is_gt_enabled(resume_id):
        try:
            from pymongo import MongoClient

            from ..overlay_mongo_labels import resolve_mongo_resume_id

            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            doc = client[mongo_db]["resumes"].find_one({"resumeId": resolve_mongo_resume_id(resume_id)})
            client.close()
            if doc:
                mongo_proj = [
                    t for t in doc.get("tokens", [])
                    if t.get("section") in ("PROJECT", "PROJECTS")
                    and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
                ]
                gt_labels, gt_segments = _build_gt_labels(mongo_proj)
        except Exception:
            pass

    scored: list[dict] = []
    correct = 0
    total = min(len(project_segments), len(pred_labels))

    for i in range(total):
        text = (project_segments[i].get("text") or "").strip()
        pred = pred_labels[i]
        gt = gt_labels[i] if i < len(gt_labels) else None
        if gt is None:
            continue
        match = gt == pred
        if match:
            correct += 1
        conf = (
            pred_confidences[i]
            if pred_confidences is not None and i < len(pred_confidences)
            else 0.0
        )
        scored.append({
            "status": "✅" if match else "❌",
            "entryKey": f"Seg {i}",
            "gt": gt,
            "pred": pred,
            "confidence": conf,
            "text": text[:140],
        })

    seg_total = len(scored)
    accuracy = (correct / seg_total * 100.0) if seg_total else 0.0

    return {
        "gtSource": "mongodb.tokens.bioLabel (segment majority → SDATE/EDATE)",
        "trainingReport": "project/phase3_segment_classification/reports/minilm/per_resume/*.md",
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
        "gtSegmentCount": len(gt_segments),
        "predSegmentCount": total,
    }
