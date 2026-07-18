"""Line-level entry divider rows (GT projectEntryHeads vs predicted B-PROJ_START lines)."""

from __future__ import annotations

from inference_v2.overlay_mongo_labels import load_mongo_entry_heads

from .entry_slice_heads import resolve_project_entry_heads


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


def build_entry_divider_line_rows(tokens: list[dict], resume_id: str) -> dict:
    """Compare MongoDB projectEntryHeads (GT) to predicted B-PROJ_START lines."""
    proj_tokens = [
        t for t in tokens
        if t.get("section") in ("PROJECT", "PROJECTS")
        and (t.get("bioLabel") or t.get("bio_label") or "O") not in ("B-HEADING", "I-HEADING")
    ]
    gt_lines = load_mongo_entry_heads(resume_id, "PROJECT")
    proj_line_keys = {(t.get("page"), t.get("lineIndex")) for t in proj_tokens}
    gt_lines &= proj_line_keys
    pred_lines = resolve_project_entry_heads(proj_tokens)

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
            "gt": "B-PROJ_START" if gt else "",
            "pred": "B-PROJ_START" if pred else "",
            "tokenLabels": _token_labels_on_line(proj_tokens, page, line_idx),
            "text": _line_text(proj_tokens, page, line_idx)[:140],
        })

    gt_total = len(gt_lines)
    fba = (matched / gt_total * 100.0) if gt_total else (100.0 if not pred_lines else 0.0)

    return {
        "gtSource": "mongodb.projectEntryHeads",
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
