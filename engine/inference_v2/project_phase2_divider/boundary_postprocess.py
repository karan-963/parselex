"""Project phase 2 — post-inference B-PROJ_START corrections (FP suppress + FN promote)."""

from __future__ import annotations

import re
from typing import Callable
from collections import Counter

from .boundary_line_rules import (
    is_link_or_date_title_line,
    is_numbered_project_title,
    is_project_metadata_line,
    should_promote_boundary,
    should_suppress_boundary,
    BULLETS,
    has_bullet,
    is_description_bullet,
    is_bullet_entry_head_line,
    is_project_title_line,
    is_continuation_fragment,
)

from .line_utils import (
    assign_line_level_token_predictions,
    collapse_boundary_to_line_anchor,
    pred_line_set_from_groups,
)


def _gap_reset(spatial_features: list, idx: int) -> bool:
    if idx >= len(spatial_features) or len(spatial_features[idx]) <= 13:
        return False
    return spatial_features[idx][13] > 0.02


def _tier_reset(seg: dict) -> bool:
    if not seg.get("spatial") or len(seg["spatial"]) <= 8:
        return False
    return float(seg["spatial"][8]) in (1.0, 2.0, 3.0)


def is_project_segment(seg: dict) -> bool:
    """Return True if the segment's majority section is PROJECT or PROJECTS."""
    seg_tokens = seg.get("tokens", [])
    if not seg_tokens:
        return False
    sections = [t.get("section", "NONE") for t in seg_tokens if t.get("section")]
    if not sections:
        return False
    majority = Counter(sections).most_common(1)[0][0]
    return majority in ("PROJECT", "PROJECTS")


def apply_project_boundary_postprocess(
    segments: list,
    seg_preds: list[str],
    spatial_features: list,
    *,
    is_project_segment: Callable = is_project_segment,
) -> list[str]:
    """Return corrected segment-level BIO labels."""
    n = len(seg_preds)
    out = list(seg_preds)
    inside_block = False

    for idx in range(n):
        seg = segments[idx]
        if not is_project_segment(seg):
            inside_block = False
            out[idx] = "O"
            continue

        text = (seg.get("text") or "").strip()
        has_b = has_bullet(text)
        numbered = bool(re.match(r"^\d+\.\s+\S", text.strip()))
        prev_b = idx > 0 and has_bullet((segments[idx - 1].get("text") or "").strip())
        transition = prev_b and not has_b
        allow_new = has_b or numbered or _tier_reset(seg) or _gap_reset(spatial_features, idx) or transition

        if out[idx] == "B-PROJ_START":
            if inside_block and not allow_new:
                out[idx] = "O"
            elif should_suppress_boundary(text) or is_continuation_fragment(text):
                out[idx] = "O"
            elif has_b and not is_project_title_line(text):
                out[idx] = "O"
            else:
                inside_block = True
        elif out[idx] == "O" and is_project_title_line(text):
            force = is_numbered_project_title(text) or is_link_or_date_title_line(text)
            if (allow_new or force) and (not inside_block or has_b or transition or force):
                out[idx] = "B-PROJ_START"
                inside_block = True
        elif out[idx] == "O" and not allow_new:
            inside_block = inside_block

    return out


def apply_metadata_boundary_prune(
    segments: list,
    seg_preds: list[str],
    *,
    is_project_segment: Callable = is_project_segment,
    line_text_by_coord: dict[tuple[int, int], str] | None = None,
) -> list[str]:
    """Final pass: strip boundaries on QA metadata / tools lines."""
    out = list(seg_preds)
    for idx, seg in enumerate(segments):
        if not is_project_segment(seg):
            continue
        text = (seg.get("text") or "").strip()
        if line_text_by_coord:
            for tok in seg.get("tokens", []):
                if tok and "page" in tok and "lineIndex" in tok:
                    key = (tok["page"], tok["lineIndex"])
                    full = line_text_by_coord.get(key, "").strip()
                    if full:
                        text = full
                        break
        if out[idx] == "B-PROJ_START" and should_suppress_boundary(text):
            out[idx] = "O"
    return out


def decode_model_seg_preds(
    preds: list,
    segments: list,
    num_segs: int,
    spatial_features: list,
    groups_eval: list,
    id2label: dict,
    *,
    use_repression: bool = False,
    is_project_segment_fn: Callable = is_project_segment,
) -> list[str]:
    """Raw model argmax → segment BIO labels (one B-PROJ_START per line group)."""
    inside_block = False
    seg_preds = ["O"] * num_segs

    for idx in range(num_segs):
        seg = segments[idx]

        if not is_project_segment_fn(seg):
            inside_block = False
            seg_preds[idx] = "O"
            continue

        was_inside = inside_block
        text = (seg.get("text") or "").strip()
        has_b = has_bullet(text)
        prev_b = idx > 0 and has_bullet((segments[idx - 1].get("text") or "").strip())
        transition = prev_b and not has_b
        allow_new = has_b or _tier_reset(seg) or _gap_reset(spatial_features, idx) or transition

        if use_repression and inside_block and not allow_new:
            repress = True
        else:
            repress = False

        model_start = id2label[preds[idx]] == "B-PROJ_START"
        if repress:
            seg_preds[idx] = "O"
        elif model_start:
            seg_preds[idx] = "B-PROJ_START"
            inside_block = True
        else:
            seg_preds[idx] = "O"
            if was_inside:
                inside_block = True

    for group in groups_eval:
        has_pred = any(seg_preds[idx] == "B-PROJ_START" for idx in group)
        for idx in group:
            if seg_preds[idx] == "B-PROJ_START":
                seg_preds[idx] = "O"
        if has_pred:
            seg_preds[group[0]] = "B-PROJ_START"

    return seg_preds


def seg_preds_to_pred_line_set(
    seg_preds: list[str],
    segments: list,
    num_segs: int,
    project_lines: set[tuple[int, int]],
    groups_eval: list[list[int]] | None = None,
) -> set[tuple[int, int]]:
    if groups_eval is not None:
        return pred_line_set_from_groups(seg_preds, groups_eval, segments, project_lines)
    tokens: list[dict] = []
    for idx in range(num_segs):
        label = seg_preds[idx]
        for token_i, tok in enumerate(segments[idx].get("tokens", [])):
            if tok and isinstance(tok, dict):
                tok = dict(tok)
                tok["prediction"] = label if token_i == 0 else "O"
                tokens.append(tok)
    return {
        (t["page"], t["lineIndex"])
        for t in tokens
        if t.get("prediction") == "B-PROJ_START"
        and "page" in t and "lineIndex" in t
        and (t["page"], t["lineIndex"]) in project_lines
    }


def apply_full_boundary_pipeline(
    preds: list,
    segments: list,
    num_segs: int,
    spatial_features: list,
    groups_eval: list,
    project_lines: set[tuple[int, int]],
    id2label: dict,
    is_project_segment_fn: Callable = is_project_segment,
    line_text_by_coord: dict[tuple[int, int], str] | None = None,
) -> tuple[list[str], set[tuple[int, int]], set[tuple[int, int]]]:
    """Return (final_seg_preds, model_line_set, post_line_set)."""
    model_seg = decode_model_seg_preds(
        preds, segments, num_segs, spatial_features, groups_eval, id2label,
        is_project_segment_fn=is_project_segment_fn
    )
    final_seg = apply_project_boundary_postprocess(
        segments[:num_segs], model_seg, spatial_features,
        is_project_segment=is_project_segment_fn,
    )
    if line_text_by_coord:
        from .style_heuristic import apply_project_style_heuristic

        final_seg = apply_project_style_heuristic(
            segments[:num_segs], final_seg, groups_eval, line_text_by_coord,
            is_project_segment=is_project_segment_fn,
        )
    final_seg = apply_metadata_boundary_prune(
        segments[:num_segs], final_seg, is_project_segment=is_project_segment_fn,
        line_text_by_coord=line_text_by_coord,
    )
    final_seg = collapse_boundary_to_line_anchor(
        final_seg, groups_eval, segments[:num_segs], is_project_segment_fn,
    )

    model_seg = collapse_boundary_to_line_anchor(
        model_seg, groups_eval, segments[:num_segs], is_project_segment_fn,
    )

    model_lines = seg_preds_to_pred_line_set(
        model_seg, segments, num_segs, project_lines, groups_eval,
    )
    post_lines = seg_preds_to_pred_line_set(
        final_seg, segments, num_segs, project_lines, groups_eval,
    )
    return final_seg, model_lines, post_lines
