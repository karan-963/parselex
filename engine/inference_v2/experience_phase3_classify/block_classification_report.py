"""Block-level field classification report (mirrors phase3 report_builder)."""

from __future__ import annotations

from collections import Counter

from ..experience_phase2_divider.heads_loader import load_entry_head_lines
from ..overlay_mongo_labels import overlay_mongo_field_labels
from .config import BIO_TO_CLASS, LABEL_LIST
from .data_utils import clean_block_text, is_punctuation_only

_CLASS_PRIORITY = ("ROLE", "COMP", "DATE", "DESC")


def _bio_to_class(bio: str) -> str | None:
    return BIO_TO_CLASS.get(bio or "O")


def _block_gt_class(block: list[dict]) -> str | None:
    """GT macro class = first B-* mongo bio in block (training _block_label).

    Reads `_fieldBioLabel` stamped by overlay_mongo_field_labels, which aligns
    MongoDB labels by line content + fuzzy match (robust to PDF re-extraction y0
    drift, unlike exact coordinate keys).
    """
    for t in block:
        bio = t.get("_fieldBioLabel") or "O"
        if bio.startswith("B-"):
            cls = _bio_to_class(bio)
            if cls:
                return cls
    counts: Counter[str] = Counter()
    for t in block:
        cls = _bio_to_class(t.get("_fieldBioLabel") or "O")
        if cls:
            counts[cls] += 1
    if not counts:
        return None
    best_cnt = counts.most_common(1)[0][1]
    tied = [lbl for lbl, cnt in counts.items() if cnt == best_cnt]
    for lbl in _CLASS_PRIORITY:
        if lbl in tied:
            return lbl
    return tied[0]


def build_block_classification_report(
    resume_id: str,
    block_rows: list[dict],
    exp_tokens: list[dict],
    slug: str | None = None,
) -> dict:
    """Build block comparison rows. Each block_row has entry_key, text, pred, block."""
    # Stamp MongoDB GT labels onto exp_tokens via content-aligned overlay. Block rows
    # reference these same token dicts, so `_block_gt_class` can read `_fieldBioLabel`.
    overlay_mongo_field_labels(exp_tokens, resume_id, slug)
    gt_head_lines = load_entry_head_lines(resume_id, exp_tokens, slug)

    scored: list[dict] = []
    correct = 0
    for row in block_rows:
        text = (row.get("text") or "").strip()
        if not text or is_punctuation_only(text):
            continue
        block = row.get("block") or []
        gt_cls = _block_gt_class(block)
        pred_cls = row.get("pred") or "DESC"
        if gt_cls is None:
            continue
        match = gt_cls == pred_cls
        if match:
            correct += 1
        scored.append({
            "status": "✅" if match else "❌",
            "entryKey": row.get("entry_key", ""),
            "gt": gt_cls,
            "pred": pred_cls,
            "confidence": row.get("confidence", 0.0),
            "text": text[:140],
        })

    total = len(scored)
    macro_f1_proxy = (correct / total * 100.0) if total else 0.0

    return {
        "gtSource": "mongodb.tokens.bioLabel (first B-* per block)",
        "entryHeadSource": "mongodb.experienceEntryHeads",
        "gtEntryHeadLines": [{"page": p, "lineIndex": l} for p, l in sorted(gt_head_lines)],
        "macroClasses": LABEL_LIST,
        "metrics": {
            "macroF1ProxyPercent": round(macro_f1_proxy, 2),
            "blocks": total,
            "correct": correct,
            "errors": total - correct,
        },
        "blockRows": scored,
    }
