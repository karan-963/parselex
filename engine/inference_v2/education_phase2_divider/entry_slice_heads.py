"""Resolve education entry head lines from step-5 boundary labels."""

from __future__ import annotations


def resolve_education_boundary_heads(tokens: list[dict]) -> set[tuple[int, int]]:
    """Lines with B-EDU_START from education_phase2_divider (step 5)."""
    heads: set[tuple[int, int]] = set()
    for t in tokens:
        label = t.get("bioLabel") or t.get("bio_label") or t.get("prediction") or "O"
        if label == "B-EDU_START":
            heads.add((
                int(t.get("page", 0)),
                int(t.get("lineIndex", t.get("line_index", 0))),
            ))
    return heads
