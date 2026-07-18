"""Education training phase 1 — token segmentation (phrase B-SEG / I-SEG).

Groups EDUCATION tokens into entries using MongoDB educationEntryHeads,
runs PhraseSegmenterTransformer per entry, applies continuity guard and
education gap/style/entity postprocess (training parity).
"""

from __future__ import annotations

import logging
from typing import Any

from inference_v2.overlay_mongo_labels import load_mongo_entry_heads, overlay_mongo_field_labels
from inference_v2.predictor_cache import get_predictor

from .data_utils import _normalize_spatial
from .entry_ops import apply_continuity_guard, description_lock_flags, group_entries
from .entry_slice_heads import resolve_education_boundary_heads, resolve_education_entry_heads
from .gap_heuristic import apply_segment_postprocess
from .metrics import build_token_segmentation_metrics
from .predictor import EducationPhase1Predictor

logger = logging.getLogger(__name__)

LABEL_MAP = ["O", "B-SEG", "I-SEG"]


def _empty_result(resume_id: str) -> dict[str, Any]:
    return {
        "stage": "education_phase1_segment",
        "title": "Education Token Segmentation",
        "section": "EDUCATION",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "education/new_phase1_token_segmentation",
        "task": "token_segmentation",
        "phraseSegmentationPipeline": "education/new_phase1_token_segmentation",
        "labelField": "prediction",
        "labels": LABEL_MAP,
        "tokenCount": 0,
        "evalTokenCount": 0,
        "nonOCount": 0,
        "sampleLabels": [],
        "tokens": [],
    }


def run_education_phase1_segment(tokens: list[dict], resume_id: str = "resume") -> dict[str, Any]:
    filtered_indices = [
        i for i, t in enumerate(tokens)
        if t.get("section") == "EDUCATION"
        and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]
    if not filtered_indices:
        return _empty_result(resume_id)

    filtered_tokens = [tokens[idx] for idx in filtered_indices]
    overlay_mongo_field_labels(filtered_tokens, resume_id)

    # Row-major order for entry slicing (column-major section sort interleaves header cells)
    filtered_tokens.sort(
        key=lambda t: (int(t.get("page", 1)), float(t.get("y0", 0)), float(t.get("x0", 0)))
    )

    spatial_all = _normalize_spatial(filtered_tokens)
    for t, spat in zip(filtered_tokens, spatial_all):
        s = list(spat)
        if len(s) < 12:
            s = s + [0.0] * (12 - len(s))
        s[11] = 0.0
        t["_temp_spatial"] = s

    edu_keys = {(t.get("page"), t.get("lineIndex")) for t in filtered_tokens}
    head_lines = resolve_education_boundary_heads(filtered_tokens) & edu_keys
    if not head_lines:
        head_lines = resolve_education_entry_heads(resume_id, filtered_tokens) & edu_keys
        if not head_lines:
            mongo_heads = load_mongo_entry_heads(resume_id, "EDUCATION")
            if mongo_heads:
                logger.warning(
                    "education_phase1_segment: no B-EDU_START lines — falling back to mongo educationEntryHeads",
                )
            else:
                logger.warning("education_phase1_segment: no educationEntryHeads — treating section as one block")
            head_lines = set()

    entries = group_entries(filtered_tokens, head_lines)
    predictor = get_predictor("education_phase1_segment", EducationPhase1Predictor)

    for _head, entry_toks in entries:
        if len(entry_toks) < 3:
            for t in entry_toks:
                t["segLabel"] = "O"
            continue

        desc_flags = description_lock_flags(entry_toks)
        for t, is_desc in zip(entry_toks, desc_flags):
            if "_temp_spatial" in t and len(t["_temp_spatial"]) >= 12:
                t["_temp_spatial"][11] = float(is_desc)

        entry_preds, entry_conf_list = predictor.segment_entry(entry_toks)
        for t, conf in zip(entry_toks, entry_conf_list):
            t["_segConfidence"] = conf
        entry_preds = apply_continuity_guard(entry_toks, entry_preds)

        desc_skip = {idx for idx, flag in enumerate(desc_flags) if flag}
        entry_preds = apply_segment_postprocess(
            entry_toks, entry_preds, skip_indices=desc_skip, use_bio_hints=True
        )

        for t, pred_id in zip(entry_toks, entry_preds):
            t["segLabel"] = LABEL_MAP[pred_id]

    for t in filtered_tokens:
        t.pop("_temp_spatial", None)
        seg = t.get("segLabel", "O")
        t["prediction"] = seg
        t["segLabel"] = seg
        t["segConfidence"] = round(float(t.pop("_segConfidence", 0.0)), 4)

    non_o_count = sum(1 for t in filtered_tokens if t.get("prediction") != "O")
    token_segmentation = build_token_segmentation_metrics(filtered_tokens)

    return {
        "stage": "education_phase1_segment",
        "title": "Education Token Segmentation",
        "section": "EDUCATION",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "education/new_phase1_token_segmentation",
        "task": "token_segmentation",
        "phraseSegmentationPipeline": "education/new_phase1_token_segmentation",
        "labelField": "prediction",
        "labels": LABEL_MAP,
        "tokenSegmentation": token_segmentation,
        "gtSegSource": "mongodb.bioLabel→B-SEG/I-SEG",
        "tokenCount": len(filtered_tokens),
        "evalTokenCount": len(filtered_tokens),
        "nonOCount": non_o_count,
        "sampleLabels": sorted({t.get("prediction", "O") for t in filtered_tokens}),
        "tokens": [
            {
                "page": t["page"],
                "lineIndex": t["lineIndex"],
                "tokenIndex": t["tokenIndex"],
                "token": t["token"],
                "prediction": t["prediction"],
                "confidence": t.get("segConfidence", 0.0),
                "x0": t.get("x0"),
                "y0": t.get("y0"),
                "x1": t.get("x1"),
                "y1": t.get("y1"),
            }
            for t in filtered_tokens
        ],
    }
