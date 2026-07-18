"""Post-process heading line sets: adjacent dedupe + phase-1 metrics (extras refinable)."""

from __future__ import annotations

from .line_builder import LineRecord


def dedupe_adjacent_heading_keys(
    lines: list[LineRecord],
    pred_keys: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Keep the first heading in each consecutive run; drop immediately following headings."""
    if not pred_keys:
        return set()
    kept: set[tuple[int, int]] = set()
    prev_heading = False
    for line in sorted(lines, key=lambda ln: ln.key):
        if line.key in pred_keys:
            if prev_heading:
                continue
            kept.add(line.key)
            prev_heading = True
        else:
            prev_heading = False
    return kept


def compute_phase1_recall(gt_lines: set[tuple[int, int]], pred_lines: set[tuple[int, int]]) -> float:
    """Recall-only score: extras do not penalize (refinable in section classification)."""
    if not gt_lines:
        return 100.0
    return len(gt_lines & pred_lines) / len(gt_lines) * 100.0


def heading_metrics(gt_lines: set[tuple[int, int]], pred_lines: set[tuple[int, int]]) -> dict[str, float | int]:
    tp = len(gt_lines & pred_lines)
    fp = len(pred_lines - gt_lines)
    fn = len(gt_lines - pred_lines)
    union = len(gt_lines | pred_lines)
    fha = (tp / union * 100.0) if union else 100.0
    recall = compute_phase1_recall(gt_lines, pred_lines)
    precision = (tp / len(pred_lines) * 100.0) if pred_lines else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fha": fha,
        "recall": recall,
        "precision": precision,
        "extras": fp,
    }
