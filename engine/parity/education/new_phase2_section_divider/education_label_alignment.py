"""Single source of truth for education boundary GT → segment label alignment."""

from __future__ import annotations

from education_boundary_line_rules import (
    is_decorative_separator_line,
    is_entry_head_candidate,
    is_section_header_line,
    is_table_header_line,
    should_suppress_boundary,
)
from education_line_utils import build_physical_line_text_map
import education_report_helpers as rh


def _sorted_education_lines(education_lines: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(education_lines, key=lambda c: (c[0], c[1]))


def _next_education_line(
    coord: tuple[int, int],
    sorted_lines: list[tuple[int, int]],
) -> tuple[int, int] | None:
    page, line = coord
    for p, l in sorted_lines:
        if p == page and l > line:
            return (p, l)
    return None


def _is_non_entry_structural_line(text: str) -> bool:
    return (
        is_section_header_line(text)
        or is_decorative_separator_line(text)
        or is_table_header_line(text)
        or should_suppress_boundary(text)
    )


def shift_heads_off_structural_lines(
    head_lines: set[tuple[int, int]],
    physical_lines: dict[tuple[int, int], str],
    education_lines: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Move GT heads off section titles / rules / table headers onto the next entry row."""
    sorted_lines = _sorted_education_lines(education_lines)
    out: set[tuple[int, int]] = set()

    for coord in sorted(head_lines, key=lambda c: (c[0], c[1])):
        text = physical_lines.get(coord, "").strip()
        if not _is_non_entry_structural_line(text):
            out.add(coord)
            continue

        nxt = _next_education_line(coord, sorted_lines)
        while nxt is not None:
            nxt_text = physical_lines.get(nxt, "").strip()
            if is_entry_head_candidate(nxt_text):
                out.add(nxt)
                break
            if not _is_non_entry_structural_line(nxt_text) and nxt_text:
                break
            nxt = _next_education_line(nxt, sorted_lines)

    return out


def resolve_education_boundary_heads(
    doc: dict,
    cleaned_tokens: list[dict],
    segments: list,
) -> set[tuple[int, int]]:
    """
    Resolved, education-scoped entry-head coordinates for training and model-only eval.
    Mirrors eval GT construction without inference post-process.
    """
    education_lines = rh.collect_education_line_coords(segments)
    raw_heads = rh.parse_entry_heads(doc, cleaned_tokens)
    scoped_heads, _ = rh.scope_heads_to_education(raw_heads, education_lines)
    if not scoped_heads:
        return set()

    physical_lines = build_physical_line_text_map(segments, cleaned_tokens)
    return shift_heads_off_structural_lines(scoped_heads, physical_lines, education_lines)


def assign_education_segment_labels(
    segments: list,
    groups: list[list[int]],
    edu_head_lines: set[tuple[int, int]],
) -> list[str]:
    """One B-EDU_START per physical entry line; inline segments on that line get I-EDU_START."""
    claimed: set[tuple[int, int]] = set()
    labels = ["O"] * len(segments)

    for group in groups:
        line_key = None
        for idx in group:
            for t in segments[idx].get("tokens", []):
                if not t or "page" not in t or "lineIndex" not in t:
                    continue
                coord = (t["page"], t["lineIndex"])
                if coord in edu_head_lines and coord not in claimed:
                    line_key = coord
                    break
            if line_key is not None:
                break

        if line_key is None:
            continue

        claimed.add(line_key)
        for pos, idx in enumerate(group):
            labels[idx] = "B-EDU_START" if pos == 0 else "I-EDU_START"

    return labels
