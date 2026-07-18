"""Education phase 2 — post-inference B-EDU_START corrections."""

from __future__ import annotations

from education_line_utils import collapse_boundary_to_line_anchor, pred_line_set_from_groups
from education_heuristic_layer import merge_promote_only
from education_style_divider import style_promoted_line_coords


def decode_model_seg_preds(
    preds: list,
    segments: list,
    num_segs: int,
    groups_eval: list,
    id2label: dict,
) -> list[str]:
    seg_preds = [id2label.get(int(p), "O") for p in preds[:num_segs]]
    for group in groups_eval:
        has_b = any(seg_preds[idx] == "B-EDU_START" for idx in group)
        for idx in group:
            if seg_preds[idx] == "B-EDU_START":
                seg_preds[idx] = "O"
        if has_b:
            anchor = next((i for i in group if segments[i].get("tokens")), group[0])
            seg_preds[anchor] = "B-EDU_START"
    return seg_preds


def apply_full_boundary_pipeline(
    preds: list,
    segments: list,
    num_segs: int,
    groups_eval: list,
    education_lines: set[tuple[int, int]],
    id2label: dict,
    is_education_segment,
    line_text_by_coord: dict[tuple[int, int], str] | None = None,
) -> tuple[list[str], set[tuple[int, int]], set[tuple[int, int]]]:
    model_seg = decode_model_seg_preds(preds, segments, num_segs, groups_eval, id2label)
    model_seg = collapse_boundary_to_line_anchor(
        model_seg, groups_eval, segments[:num_segs], is_education_segment,
    )
    model_lines = pred_line_set_from_groups(
        model_seg, groups_eval, segments[:num_segs], education_lines,
    )

    # Style rescue only when the model predicts no education boundaries (Abhay-style misses).
    # Partial-model resumes are left untouched so post-process never drags aggregate FBA down.
    post_lines = model_lines
    if line_text_by_coord and not model_lines:
        extra_lines = style_promoted_line_coords(
            segments[:num_segs], model_seg, groups_eval, line_text_by_coord,
            is_education_segment=is_education_segment,
        )
        extra_lines &= education_lines
        post_lines = merge_promote_only(model_lines, extra_lines)

    return model_seg, model_lines, post_lines
