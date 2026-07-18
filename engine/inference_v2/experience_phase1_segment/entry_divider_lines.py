"""Line-level entry divider rows (GT from experienceEntryHeads vs predicted B-ENTRY lines)."""

from __future__ import annotations

from ..experience_phase2_divider.heads_loader import load_entry_head_lines
from .entry_slice_heads import resolve_entry_slice_heads


def _line_key(token: dict) -> tuple[int, int]:
    return (int(token.get("page", 0)), int(token.get("lineIndex", token.get("line_index", 0))))


def _pred_entry_lines(tokens: list[dict]) -> set[tuple[int, int]]:
    """Predicted entry divider lines used for step-9 segmentation (primary title heads)."""
    return resolve_entry_slice_heads(tokens)


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


def build_entry_divider_line_rows(
    tokens: list[dict],
    resume_id: str,
    *,
    pred_tokens: list[dict] | None = None,
) -> dict:
    """
    Compare MongoDB experienceEntryHeads (GT divider lines) to predicted B-ENTRY lines.

    ``pred_tokens`` — optional step-8 boundary artifact tokens; when provided, predicted
    divider lines are taken from artifact predictions instead of ``bioLabel`` on ``tokens``.
    """
    gt_lines = load_entry_head_lines(resume_id, tokens)
    exp_tokens = [
        t for t in tokens
        if t.get("section") == "EXPERIENCE"
        and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]
    if pred_tokens:
        # Replay step-8 labels onto a copy for slice-head resolution.
        replay_tokens = []
        bnd_by = {
            (t["page"], t["lineIndex"], t["tokenIndex"]): t.get("prediction", "O")
            for t in pred_tokens
        }
        for t in exp_tokens:
            key = (t.get("page"), t.get("lineIndex"), t.get("tokenIndex"))
            replay = dict(t)
            if key in bnd_by:
                replay["bioLabel"] = bnd_by[key]
                replay["bio_label"] = bnd_by[key]
            replay_tokens.append(replay)
        pred_lines = resolve_entry_slice_heads(replay_tokens)
    else:
        pred_lines = _pred_entry_lines(exp_tokens)

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
            "gt": "B-ENTRY" if gt else "",
            "pred": "B-ENTRY" if pred else "",
            "tokenLabels": _token_labels_on_line(exp_tokens, page, line_idx),
            "text": _line_text(exp_tokens, page, line_idx)[:140],
        })

    gt_total = len(gt_lines)
    fba = (matched / gt_total * 100.0) if gt_total else (100.0 if not pred_lines else 0.0)

    return {
        "gtSource": "mongodb.experienceEntryHeads",
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
