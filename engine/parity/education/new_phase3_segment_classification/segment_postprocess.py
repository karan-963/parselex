"""Spatial block heuristics for education segment classification."""

from __future__ import annotations

import re

from segment_context import degree_likeness_score, institution_likeness_score
from segment_label_rules import is_decorative_segment, refine_segment_label, should_downgrade_institution

_BULLET_PREFIX = re.compile(r"^[•\-\*●·▪■◦✓✔\uf0b7\uf0a7\s]+")
_TRIVIAL_SEGMENT = re.compile(r"^[\|–—\-,\s\.]+$")

DEFAULT_X_ALIGN_MARGIN = 10.0
MAX_HEADING_WORDS = 10

_DEGREE_ABBR = re.compile(
    r"\b(?:b\.?\s*tech|b\.?\s*e\.?|m\.?\s*tech|mba|bca|mca|b\.?\s*sc|m\.?\s*sc|diploma)\b",
    re.IGNORECASE,
)


def _seg_layout(seg: dict) -> dict:
    sp = seg.get("spatial") or []
    toks = seg.get("tokens") or []
    if toks:
        x0 = min(float(t.get("x0", 0.0)) for t in toks)
        y0 = min(float(t.get("y0", 0.0)) for t in toks)
    else:
        x0 = float(sp[5]) if len(sp) > 5 else 0.0
        y0 = float(sp[4]) if len(sp) > 4 else 0.0
    return {
        "fs": float(sp[0]) if sp else 9.0,
        "bold": bool(sp[1]) if len(sp) > 1 else False,
        "page": float(sp[3]) if len(sp) > 3 else 1.0,
        "x0": x0,
        "y0": y0,
    }


def _starts_with_bullet(text: str) -> bool:
    return bool(_BULLET_PREFIX.match(text.strip()))


def _is_trivial_segment(text: str) -> bool:
    clean = text.strip()
    return not clean or bool(_TRIVIAL_SEGMENT.match(clean)) or len(clean) <= 2


def _should_promote_institution(seg: dict, text: str, model_label: str) -> bool:
    if should_downgrade_institution(text) or is_decorative_segment(text):
        return False
    layout = _seg_layout(seg)
    if model_label == "INSTITUTION":
        return True
    return layout["bold"] and institution_likeness_score(text) >= 0.35


def _should_promote_degree(text: str, model_label: str) -> bool:
    if model_label == "DEGREE":
        return True
    return degree_likeness_score(text) >= 0.45 or bool(_DEGREE_ABBR.search(text))


def apply_education_block_heuristics(
    education_segments: list[dict],
    labels: list[str],
) -> list[str]:
    if not education_segments or not labels:
        return list(labels)

    out = list(labels)
    anchor_x0: float | None = None

    for i, seg in enumerate(education_segments):
        text = seg.get("text", "").strip()
        if _is_trivial_segment(text) or is_decorative_segment(text):
            continue

        layout = _seg_layout(seg)

        if out[i] == "DATE":
            continue

        if _should_promote_institution(seg, text, out[i]):
            if anchor_x0 is None or abs(layout["x0"] - anchor_x0) <= DEFAULT_X_ALIGN_MARGIN:
                out[i] = "INSTITUTION"
                if layout["bold"] and institution_likeness_score(text) >= 0.4:
                    anchor_x0 = layout["x0"]
            continue

        if _should_promote_degree(text, out[i]) and not _starts_with_bullet(text):
            out[i] = "DEGREE"

    return out


def postprocess_segment_predictions(
    education_segments: list[dict],
    labels: list[str],
) -> list[str]:
    adjusted = apply_education_block_heuristics(education_segments, labels)
    return [
        refine_segment_label(education_segments[i].get("text", "").strip(), lbl)
        for i, lbl in enumerate(adjusted)
    ]
