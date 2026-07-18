"""Spatial block heuristics for project segment classification.

Project entries are grouped by bold title lines aligned to the resume's
project-title left margin. Each group opener is promoted to PROJECT_NAME;
indented or pipe-stack lines are never treated as new project headings.
"""

from __future__ import annotations

import re

from segment_label_rules import (
    is_decorative_segment,
    refine_segment_label,
    should_downgrade_project_name,
)

_BULLET_PREFIX = re.compile(r"^[•\-\*●·▪■◦✓✔\uf0b7\uf0a7\s]+")
_TRIVIAL_SEGMENT = re.compile(r"^[\|–—\-,\s\.]+$")
_NUMBERED_SUBENTRY = re.compile(r"^\d+\.\s+")

DEFAULT_X_ALIGN_MARGIN = 10.0
DEFAULT_FS_TOLERANCE = 0.5
MAX_HEADING_WORDS = 12


def _seg_layout(seg: dict) -> dict:
    sp = seg.get("spatial") or []
    toks = seg.get("tokens") or []
    if toks:
        x0 = min(float(t.get("x0", 0.0)) for t in toks)
        y0 = min(float(t.get("y0", 0.0)) for t in toks)
        y1 = max(float(t.get("y1", 0.0)) for t in toks)
    else:
        x0 = float(sp[5]) if len(sp) > 5 else 0.0
        y0 = float(sp[4]) if len(sp) > 4 else 0.0
        y1 = float(sp[6]) if len(sp) > 6 else y0
    return {
        "fs": float(sp[0]) if sp else 9.0,
        "bold": bool(sp[1]) if len(sp) > 1 else False,
        "page": float(sp[3]) if len(sp) > 3 else 1.0,
        "tier": float(sp[8]) if len(sp) > 8 else 0.0,
        "x0": x0,
        "y0": y0,
        "y1": y1,
    }


def _starts_with_bullet(text: str) -> bool:
    return bool(_BULLET_PREFIX.match(text.strip()))


def _is_trivial_segment(text: str) -> bool:
    clean = text.strip()
    return not clean or bool(_TRIVIAL_SEGMENT.match(clean)) or len(clean) <= 2


def _style_matches_anchor(
    anchor: dict,
    layout: dict,
    *,
    x_margin: float,
    fs_tolerance: float,
) -> bool:
    if int(anchor["page"]) != int(layout["page"]):
        return False
    if abs(anchor["fs"] - layout["fs"]) > fs_tolerance:
        return False
    if anchor["bold"] != layout["bold"]:
        return False
    return abs(anchor["x0"] - layout["x0"]) <= x_margin


def _is_date_label(label: str) -> bool:
    return label in ("DATE", "SDATE", "EDATE")


def _title_anchor_x0(
    project_segments: list[dict],
    before_idx: int,
    *,
    x_margin: float,
) -> float | None:
    """Most recent left-aligned bold project-title x0 before ``before_idx``."""
    for i in range(before_idx - 1, -1, -1):
        seg = project_segments[i]
        text = seg.get("text", "").strip()
        if is_decorative_segment(text) or should_downgrade_project_name(text):
            continue
        layout = _seg_layout(seg)
        if not layout["bold"]:
            continue
        words = text.split()
        if len(words) == 1 and not (text.isupper() and len(text) >= 4):
            continue
        if len(words) > MAX_HEADING_WORDS:
            continue
        return layout["x0"]
    return None


def _is_new_project_block_start(
    seg: dict,
    text: str,
    prev_seg: dict | None = None,
    *,
    project_segments: list[dict] | None = None,
    seg_idx: int = 0,
    x_margin: float = DEFAULT_X_ALIGN_MARGIN,
) -> bool:
    """A new project block begins only at a bold, left-aligned title line."""
    clean = text.strip()
    if not clean or _is_trivial_segment(clean) or is_decorative_segment(clean):
        return False
    if _starts_with_bullet(clean) or should_downgrade_project_name(clean):
        return False
    if _NUMBERED_SUBENTRY.match(clean):
        return False

    layout = _seg_layout(seg)
    if not layout["bold"]:
        return False

    words = clean.split()
    if len(words) == 1:
        if not (clean.isupper() and len(clean) >= 4):
            return False
    elif len(words) > MAX_HEADING_WORDS:
        return False

    if prev_seg is not None:
        prev_text = prev_seg.get("text", "").strip()
        if _is_trivial_segment(prev_text) or is_decorative_segment(prev_text):
            return False
        pl = _seg_layout(prev_seg)
        if (
            int(pl["page"]) == int(layout["page"])
            and abs(layout["y0"] - pl["y0"]) <= 2.0
        ):
            return False

    if project_segments is not None:
        anchor_x0 = _title_anchor_x0(
            project_segments, seg_idx, x_margin=x_margin
        )
        if anchor_x0 is not None and abs(layout["x0"] - anchor_x0) > x_margin:
            return False

    return True


def _should_promote_opener(seg: dict, text: str, model_label: str) -> bool:
    if is_decorative_segment(text) or should_downgrade_project_name(text):
        return False
    layout = _seg_layout(seg)
    if model_label == "PROJECT_NAME":
        return True
    return layout["bold"] and len(text.split()) <= MAX_HEADING_WORDS


def _should_promote_style_match(seg: dict, text: str, model_label: str) -> bool:
    if should_downgrade_project_name(text):
        return False
    layout = _seg_layout(seg)
    if model_label == "PROJECT_NAME":
        return True
    return layout["bold"] and len(text.split()) <= 8


def _skip_before_first_title(seg: dict) -> bool:
    text = seg.get("text", "").strip()
    return is_decorative_segment(text) or _is_trivial_segment(text)


def _iter_project_blocks(
    project_segments: list[dict],
    *,
    x_margin: float = DEFAULT_X_ALIGN_MARGIN,
) -> list[tuple[int, int]]:
    """Return [start, end) indices for each bold-title project block."""
    if not project_segments:
        return []

    start = 0
    while start < len(project_segments) and _skip_before_first_title(
        project_segments[start]
    ):
        start += 1
    if start >= len(project_segments):
        return []

    blocks: list[tuple[int, int]] = []
    block_begin = start
    for idx in range(start + 1, len(project_segments)):
        text = project_segments[idx].get("text", "").strip()
        if _is_new_project_block_start(
            project_segments[idx],
            text,
            project_segments[idx - 1],
            project_segments=project_segments,
            seg_idx=idx,
            x_margin=x_margin,
        ):
            blocks.append((block_begin, idx))
            block_begin = idx
    blocks.append((block_begin, len(project_segments)))
    return blocks


def apply_project_block_heuristics(
    project_segments: list[dict],
    labels: list[str],
    *,
    x_margin: float = DEFAULT_X_ALIGN_MARGIN,
    fs_tolerance: float = DEFAULT_FS_TOLERANCE,
    **_,
) -> list[str]:
    """Promote bold block openers and style-aligned sub-headings."""
    if not project_segments or not labels:
        return list(labels)

    out = list(labels)
    for start, end in _iter_project_blocks(project_segments, x_margin=x_margin):
        anchor = _seg_layout(project_segments[start])
        opener_text = project_segments[start].get("text", "").strip()
        anchor_promoted = False

        if not _is_date_label(out[start]) and _should_promote_opener(
            project_segments[start], opener_text, out[start]
        ):
            out[start] = "PROJECT_NAME"
            anchor_promoted = True
        elif (
            out[start] == "PROJECT_NAME"
            and not should_downgrade_project_name(opener_text)
            and not is_decorative_segment(opener_text)
        ):
            anchor_promoted = True

        if not anchor_promoted:
            continue

        for k in range(start + 1, end):
            if _is_date_label(out[k]):
                continue

            seg = project_segments[k]
            text = seg.get("text", "").strip()
            layout = _seg_layout(seg)
            if not _style_matches_anchor(
                anchor, layout, x_margin=x_margin, fs_tolerance=fs_tolerance
            ):
                continue
            if _should_promote_style_match(seg, text, out[k]):
                out[k] = "PROJECT_NAME"

    return out


def postprocess_segment_predictions(
    project_segments: list[dict],
    labels: list[str],
    **kwargs,
) -> list[str]:
    """Block heuristics + per-segment text rules."""
    adjusted = apply_project_block_heuristics(project_segments, labels, **kwargs)
    return [
        refine_segment_label(project_segments[i].get("text", "").strip(), lbl)
        for i, lbl in enumerate(adjusted)
    ]
