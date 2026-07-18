"""Line-level entry divider rows (GT educationEntryHeads vs predicted B-EDU_START lines)."""

from __future__ import annotations

from inference_v2.overlay_mongo_labels import load_mongo_entry_heads

from .entry_slice_heads import resolve_education_boundary_heads
from .gt_heads import resolve_education_gt_heads
from .y0_line_collapse import collapse_lines_by_y0


def _line_text(tokens: list[dict], page: int, line_index: int) -> str:
    parts = [
        t.get("token", "")
        for t in tokens
        if int(t.get("page", 0)) == page and int(t.get("lineIndex", t.get("line_index", 0))) == line_index
    ]
    return " ".join(parts).strip()


def _token_labels_on_line(tokens: list[dict], page: int, line_index: int) -> str:
    labels = sorted(
        {
            t.get("bioLabel") or t.get("bio_label") or t.get("prediction") or "O"
            for t in tokens
            if int(t.get("page", 0)) == page and int(t.get("lineIndex", t.get("line_index", 0))) == line_index
        }
    )
    return ", ".join(labels)


def _resolve_gt_heads(edu_tokens: list[dict], resume_id: str) -> set[tuple[int, int]]:
    return resolve_education_gt_heads(resume_id, edu_tokens)


def build_entry_divider_line_rows(tokens: list[dict], resume_id: str) -> dict:
    """Compare MongoDB educationEntryHeads (GT) to predicted B-EDU_START lines."""
    edu_tokens = [
        t for t in tokens
        if t.get("section") == "EDUCATION"
        and (t.get("bioLabel") or t.get("bio_label") or "O") not in ("B-HEADING", "I-HEADING")
    ]
    edu_line_keys = {(t.get("page"), t.get("lineIndex")) for t in edu_tokens}
    gt_training = _resolve_gt_heads(edu_tokens, resume_id)
    gt_mongo_raw = load_mongo_entry_heads(resume_id, "EDUCATION")
    gt_lines = gt_training & edu_line_keys
    if not gt_lines:
        gt_lines = gt_mongo_raw & edu_line_keys

    pred_lines_raw = resolve_education_boundary_heads(edu_tokens)
    pred_lines = collapse_lines_by_y0(edu_tokens, pred_lines_raw)

    all_keys = sorted(gt_lines | pred_lines, key=lambda k: (k[0], k[1]))
    rows: list[dict] = []
    matched = missed = extra = 0

    for page, line_idx in all_keys:
        gt = (page, line_idx) in gt_lines
        pred = (page, line_idx) in pred_lines
        if gt and pred:
            status = "✅ MATCH"
            matched += 1
        elif gt:
            status = "❌ MISSED"
            missed += 1
        else:
            status = "⚠️ EXTRA"
            extra += 1

        rows.append({
            "status": status,
            "page": page,
            "line": line_idx,
            "gt": "B-EDU_START" if gt else "",
            "pred": "B-EDU_START" if pred else "",
            "tokenLabels": _token_labels_on_line(edu_tokens, page, line_idx),
            "text": _line_text(edu_tokens, page, line_idx)[:140],
        })

    gt_total = len(gt_lines)
    fba = (matched / gt_total * 100.0) if gt_total else (100.0 if not pred_lines else 0.0)

    return {
        "gtSource": "mongodb.educationEntryHeads",
        "gtEntryLines": [{"page": p, "lineIndex": l} for p, l in sorted(gt_lines)],
        "predEntryLines": [{"page": p, "lineIndex": l} for p, l in sorted(pred_lines)],
        "metrics": {
            "fbaPercent": round(fba, 2),
            "gtEntryLines": gt_total,
            "matched": matched,
            "missed": missed,
            "extra": extra,
        },
        "lineRows": rows,
    }
