"""Per-line spatial and context features for line classifiers."""

from __future__ import annotations

from typing import Any

from .line_builder import LineRecord

SPATIAL_DIM = 10
MAX_CONTEXT_CHARS = 120


def _norm(val: float, lo: float, hi: float) -> float:
    span = max(hi - lo, 1e-6)
    return max(0.0, min(1.0, (val - lo) / span))


def extract_line_spatial(
    line: LineRecord,
    all_lines: list[LineRecord],
    page_height: float = 792.0,
    page_width: float = 612.0,
    gap_to_prev: float = 0.0,
) -> list[float]:
    """10D normalized spatial vector for one line."""
    idx = all_lines.index(line) if line in all_lines else 0
    line_rank = idx / max(len(all_lines) - 1, 1)
    width = line.x1 - line.x0
    return [
        _norm(line.x0, 0, page_width),
        _norm(line.y0, 0, page_height),
        _norm(width, 0, page_width),
        _norm(line.font_median, 8.0, 18.0),
        float(line.is_bold),
        float(line.is_caps),
        line_rank,
        _norm(gap_to_prev, 0, 30.0),
        min(len(line.text.split()) / 15.0, 1.0),
        float(line.page) / 10.0,
    ]


def build_line_samples(
    tokens: list[dict[str, Any]],
    lines: list[LineRecord],
) -> list[dict[str, Any]]:
    """Build feature dicts for each line with prev/next context."""
    if not lines:
        return []

    xs = [t.get("x0", 0.0) for t in tokens]
    ys = [t.get("y0", 0.0) for t in tokens]
    x1s = [t.get("x1", 0.0) for t in tokens]
    y1s = [t.get("y1", 0.0) for t in tokens]
    pw = max(max(x1s) - min(xs), 1.0) if xs else 612.0
    ph = max(max(y1s) - min(ys), 1.0) if ys else 792.0

    samples: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        prev_text = lines[i - 1].text[:MAX_CONTEXT_CHARS] if i > 0 else ""
        next_text = lines[i + 1].text[:MAX_CONTEXT_CHARS] if i < len(lines) - 1 else ""
        gap_prev = 0.0
        if i > 0 and lines[i - 1].page == line.page:
            gap_prev = max(line.y0 - lines[i - 1].y1, 0.0)

        is_heading = any(tokens[j].get("bioLabelId", 0) in (1, 2) for j in line.token_indices)
        spatial = extract_line_spatial(line, lines, ph, pw, gap_prev)

        samples.append({
            "key": line.key,
            "text": line.text,
            "prev_text": prev_text,
            "next_text": next_text,
            "spatial": spatial,
            "bbox": [line.x0, line.y0, line.x1, line.y1],
            "label": 1 if is_heading else 0,
            "token_indices": line.token_indices,
        })
    return samples
