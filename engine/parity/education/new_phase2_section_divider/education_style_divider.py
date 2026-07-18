"""Promote-only style rhythm post-process for education entry boundaries."""

from __future__ import annotations

from collections import Counter

from education_boundary_line_rules import (
    is_decorative_separator_line,
    is_description_bullet,
    is_degree_opener_line,
    is_degree_qualification_line,
    is_education_metadata_line,
    is_entry_head_candidate,
    is_institution_entry_line,
    is_section_header_line,
    is_table_header_line,
)
from education_line_utils import line_keys_for_group


def _line_text(group: list[int], segments: list, line_text_by_coord: dict) -> str:
    parts: list[str] = []
    for idx in group:
        for t in segments[idx].get("tokens", []) or []:
            tok = (t.get("token") or "").strip()
            if tok:
                parts.append(tok)
    if parts:
        return " ".join(parts)
    for key in line_keys_for_group(group, segments):
        if key in line_text_by_coord and line_text_by_coord[key]:
            return line_text_by_coord[key]
    return segments[group[0]].get("text", "")


def get_line_style(group: list[int], segments: list) -> tuple[float, bool, float] | None:
    tokens = []
    for idx in group:
        tokens.extend(segments[idx].get("tokens", []) or [])
    if not tokens:
        return None
    sizes = [round(float(t.get("fontSize", 9.0)), 1) for t in tokens]
    bold = sum(1 for t in tokens if t.get("isBold")) > len(tokens) / 2
    x0 = min(float(t.get("x0", 0.0)) for t in tokens)
    return (Counter(sizes).most_common(1)[0][0], bold, x0)


def _is_skippable_line(text: str) -> bool:
    return (
        is_section_header_line(text)
        or is_decorative_separator_line(text)
        or is_table_header_line(text)
        or is_description_bullet(text)
        or is_education_metadata_line(text)
    )


def _style_matches(
    curr: tuple[float, bool, float],
    anchor: tuple[float, bool, float],
    *,
    require_bold: bool,
) -> bool:
    size_c, bold_c, x0_c = curr
    size_a, bold_a, x0_a = anchor
    if abs(size_c - size_a) > 0.5 or abs(x0_c - x0_a) > 12.0:
        return False
    if require_bold:
        return bold_c and bold_a
    return True


def _is_institution_name_line(text: str) -> bool:
    if _is_skippable_line(text):
        return False
    lower = text.strip().lower()
    institution_keys = (
        "university", "college", "institute", "institution", "school", "academy",
        "polytechnic", "board", "campus",
    )
    if not any(k in lower for k in institution_keys):
        return False
    if is_degree_opener_line(text) or is_degree_qualification_line(text):
        return False
    return True


def _is_institution_detail_line(text: str) -> bool:
    if not _is_institution_name_line(text):
        return False
    return is_institution_entry_line(text)


def _is_degree_detail_line(text: str) -> bool:
    if not (is_degree_opener_line(text) or is_degree_qualification_line(text)):
        return False
    if _is_institution_name_line(text):
        return False
    return True


def _is_split_institution_opener(text: str, style: tuple[float, bool, float]) -> bool:
    return style[1] and _is_institution_name_line(text) and not is_institution_entry_line(text)


def _is_first_entry_line(text: str, style: tuple[float, bool, float]) -> bool:
    if is_entry_head_candidate(text):
        return True
    return style[1] and _is_institution_name_line(text)


def _is_next_entry_divider(
    text: str,
    style: tuple[float, bool, float],
    anchor: tuple[float, bool, float],
    rhythm: str,
) -> bool:
    if not _style_matches(style, anchor, require_bold=(rhythm == "bold")):
        return False
    if rhythm == "bold" and not style[1]:
        return False
    if is_entry_head_candidate(text):
        return True
    if rhythm == "bold" and _is_split_institution_opener(text, style):
        return True
    return False


def _education_groups(
    groups: list[list[int]],
    segments: list,
    *,
    is_education_segment,
) -> list[list[int]]:
    out: list[list[int]] = []
    for group in groups:
        if any(is_education_segment(segments[idx]) for idx in group):
            out.append(group)
    return out


def _next_valid_group(
    edu_groups: list[list[int]],
    current: list[int],
    segments: list,
    line_text_by_coord: dict[tuple[int, int], str],
) -> list[int] | None:
    try:
        idx = edu_groups.index(current)
    except ValueError:
        return None
    for group in edu_groups[idx + 1:]:
        text = _line_text(group, segments, line_text_by_coord).strip()
        if _is_skippable_line(text):
            continue
        return group
    return None


def apply_education_style_divider_heuristic(
    segments: list,
    seg_preds: list[str],
    groups: list[list[int]],
    line_text_by_coord: dict[tuple[int, int], str],
    *,
    is_education_segment,
) -> list[str]:
    """
    Promote-only style rhythm for missed entry boundaries (model predicted none).

    Bold rhythm: bold entry → non-bold detail → next bold match = new entry.
    Size rhythm: larger entry → smaller detail → next matching size = new entry.
    Split pattern: bold institution (no date) → non-bold degree on next line.
    """
    out = list(seg_preds)
    edu_groups = _education_groups(groups, segments, is_education_segment=is_education_segment)
    if not edu_groups:
        return out

    anchor: tuple[float, bool, float] | None = None
    rhythm: str | None = None
    phase = "seek_first"
    saw_detail = False
    prev_was_split_institution = False

    for group in edu_groups:
        text = _line_text(group, segments, line_text_by_coord).strip()
        if _is_skippable_line(text):
            continue

        style = get_line_style(group, segments)
        if style is None:
            continue

        first_idx = group[0]
        promote = False

        if phase == "seek_first":
            if _is_first_entry_line(text, style):
                anchor = style
                promote = True
                next_group = _next_valid_group(edu_groups, group, segments, line_text_by_coord)
                if next_group is not None:
                    nxt_style = get_line_style(next_group, segments)
                    if nxt_style and nxt_style[0] < style[0] - 0.3:
                        rhythm = "size"
                if rhythm is None and style[1]:
                    rhythm = "bold"
                elif rhythm is None:
                    rhythm = "size"
                phase = "in_entry"
                saw_detail = False
                prev_was_split_institution = _is_split_institution_opener(text, style)
        elif anchor is not None and rhythm is not None:
            if prev_was_split_institution and not style[1]:
                if is_degree_opener_line(text) or is_degree_qualification_line(text):
                    if _style_matches(style, anchor, require_bold=False):
                        promote = True
                saw_detail = True
                prev_was_split_institution = False
            elif _is_degree_detail_line(text) or _is_institution_detail_line(text):
                saw_detail = True
            elif saw_detail and _is_next_entry_divider(text, style, anchor, rhythm):
                promote = True
                anchor = style
                saw_detail = False
                prev_was_split_institution = _is_split_institution_opener(text, style)

        if promote and out[first_idx] != "B-EDU_START":
            out[first_idx] = "B-EDU_START"

    return out


def style_promoted_line_coords(
    segments: list,
    seg_preds: list[str],
    groups: list[list[int]],
    line_text_by_coord: dict[tuple[int, int], str],
    *,
    is_education_segment,
) -> set[tuple[int, int]]:
    styled = apply_education_style_divider_heuristic(
        segments, seg_preds, groups, line_text_by_coord,
        is_education_segment=is_education_segment,
    )
    coords: set[tuple[int, int]] = set()
    for group in groups:
        if styled[group[0]] != "B-EDU_START":
            continue
        if seg_preds[group[0]] == "B-EDU_START":
            continue
        if not any(is_education_segment(segments[idx]) for idx in group):
            continue
        coords.update(line_keys_for_group(group, segments))
    return coords
