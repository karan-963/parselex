"""Line-neighbor layout signals for education spatial features (train + infer)."""

from __future__ import annotations

from education_boundary_line_rules import (
    is_degree_opener_line,
    is_degree_qualification_line,
    is_split_institution_line,
)


def segment_line_coord(seg: dict) -> tuple[int, int] | None:
    for t in seg.get("tokens", []):
        if t and "page" in t and "lineIndex" in t:
            return (int(t["page"]), int(t["lineIndex"]))
    return None


def _sorted_page_lines(
    line_text_by_coord: dict[tuple[int, int], str],
    page: int,
) -> list[tuple[int, int]]:
    return sorted(
        (p, l) for p, l in line_text_by_coord if p == page
    )


def _neighbor_line_text(
    coord: tuple[int, int],
    line_text_by_coord: dict[tuple[int, int], str],
    direction: int,
) -> str:
    page, line = coord
    lines = _sorted_page_lines(line_text_by_coord, page)
    idx = next((i for i, c in enumerate(lines) if c == coord), None)
    if idx is None:
        return ""
    neighbor_idx = idx + direction
    if neighbor_idx < 0 or neighbor_idx >= len(lines):
        return ""
    return line_text_by_coord.get(lines[neighbor_idx], "").strip()


def layout_context_dims(
    seg_idx: int,
    segments: list,
    line_text_by_coord: dict[tuple[int, int], str],
) -> list[float]:
    """
    Two dims encoding split-institution / degree-below layout (Aakash pattern).
    Computed from physical line text — same at train and inference.
    """
    coord = segment_line_coord(segments[seg_idx])
    if coord is None or not line_text_by_coord:
        return [0.0, 0.0]

    prev_text = _neighbor_line_text(coord, line_text_by_coord, -1)
    next_text = _neighbor_line_text(coord, line_text_by_coord, 1)
    curr_text = line_text_by_coord.get(coord, "").strip()

    prev_split_inst = 1.0 if prev_text and is_split_institution_line(prev_text) else 0.0
    next_degree = 1.0 if next_text and (
        is_degree_opener_line(next_text) or is_degree_qualification_line(next_text)
    ) else 0.0
    on_split_inst = 1.0 if curr_text and is_split_institution_line(curr_text) else 0.0
    return [prev_split_inst, next_degree, on_split_inst]