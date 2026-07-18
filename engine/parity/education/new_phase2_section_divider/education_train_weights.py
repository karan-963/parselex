"""Per-segment loss weights for institution-first GT heads (training only)."""

from __future__ import annotations

from education_boundary_line_rules import is_split_institution_line
from education_layout_features import segment_line_coord

INSTITUTION_ONLY_B_WEIGHT = 3.0
DEGREE_BELOW_INSTITUTION_O_WEIGHT = 2.0


def build_segment_loss_weights(
    segments: list,
    groups: list[list[int]],
    segment_labels: list[str],
    line_text_by_coord: dict[tuple[int, int], str],
) -> list[float]:
    """
    Upweight B-EDU_START on split-institution rows; upweight O on degree row below
  when the institution line above is the labeled head (Aakash layout).
    """
    weights = [1.0] * len(segments)
    head_lines: set[tuple[int, int]] = set()

    for group in groups:
        for idx in group:
            if segment_labels[idx] == "B-EDU_START":
                coord = segment_line_coord(segments[idx])
                if coord:
                    head_lines.add(coord)
                text = line_text_by_coord.get(coord, "") if coord else ""
                if coord and is_split_institution_line(text):
                    weights[idx] = INSTITUTION_ONLY_B_WEIGHT
                break

    page_lines: dict[int, list[tuple[int, int]]] = {}
    for coord in line_text_by_coord:
        page_lines.setdefault(coord[0], []).append(coord)
    for page in page_lines:
        page_lines[page].sort(key=lambda c: c[1])

    for page, lines in page_lines.items():
        for i, coord in enumerate(lines):
            if i == 0:
                continue
            prev = lines[i - 1]
            if prev not in head_lines:
                continue
            prev_text = line_text_by_coord.get(prev, "")
            if not is_split_institution_line(prev_text):
                continue
            for group in groups:
                for idx in group:
                    if segment_line_coord(segments[idx]) != coord:
                        continue
                    if segment_labels[idx] == "O":
                        weights[idx] = max(weights[idx], DEGREE_BELOW_INSTITUTION_O_WEIGHT)
    return weights
