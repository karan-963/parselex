"""Parser-line records and ground-truth heading keys for section Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass
class LineRecord:
    page: int
    lineIndex: int
    key: tuple[int, int]
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_median: float
    is_bold: bool
    is_caps: bool
    token_indices: list[int]
    source: str = "parser"


def _line_stats(tokens: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    subset = sorted((tokens[i] for i in indices), key=lambda t: t.get("x0", 0.0))
    text = " ".join(str(t.get("token", "")).strip() for t in subset).strip()
    x0 = min(t.get("x0", 0.0) for t in subset) if subset else 0.0
    y0 = min(t.get("y0", 0.0) for t in subset) if subset else 0.0
    x1 = max(t.get("x1", 0.0) for t in subset) if subset else 0.0
    y1 = max(t.get("y1", 0.0) for t in subset) if subset else 0.0
    fonts = [float(t.get("fontSize", t.get("font_size", 11.0)) or 11.0) for t in subset]
    font_median = median(fonts) if fonts else 11.0
    first = subset[0] if subset else {}
    is_bold = bool(first.get("isBold", first.get("is_bold", False)))
    is_caps = len(text) > 1 and text == text.upper()
    return {
        "text": text,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "font_median": font_median,
        "is_bold": is_bold,
        "is_caps": is_caps,
    }


def build_parser_lines(tokens: list[dict[str, Any]]) -> list[LineRecord]:
    line_map: dict[tuple[int, int], list[int]] = {}
    for i, t in enumerate(tokens):
        page = int(t.get("page", 0))
        li = int(t.get("lineIndex", t.get("line_index", 0)))
        key = (page, li)
        line_map.setdefault(key, []).append(i)

    lines: list[LineRecord] = []
    for key in sorted(line_map.keys()):
        indices = line_map[key]
        stats = _line_stats(tokens, indices)
        if not stats["text"]:
            continue
        page, li = key
        lines.append(
            LineRecord(
                page=page,
                lineIndex=li,
                key=key,
                text=stats["text"],
                x0=stats["x0"],
                y0=stats["y0"],
                x1=stats["x1"],
                y1=stats["y1"],
                font_median=stats["font_median"],
                is_bold=stats["is_bold"],
                is_caps=stats["is_caps"],
                token_indices=indices,
            )
        )
    return lines


def _bio_is_heading(token: dict[str, Any]) -> bool:
    label = token.get("bioLabel", token.get("bio_label", "O"))
    if isinstance(label, str):
        return label in {"B-HEADING", "I-HEADING"}
    bio_id = token.get("bioLabelId", 0)
    return bio_id in (1, 2)


def gt_heading_keys(tokens: list[dict[str, Any]]) -> set[tuple[int, int]]:
    lines = build_parser_lines(tokens)
    gt: set[tuple[int, int]] = set()
    for line in lines:
        if any(_bio_is_heading(tokens[j]) for j in line.token_indices):
            gt.add(line.key)
    return gt


def run_alignment_audit(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    """Stub for legacy bakeoff imports."""
    return {"lines": len(build_parser_lines(tokens))}
