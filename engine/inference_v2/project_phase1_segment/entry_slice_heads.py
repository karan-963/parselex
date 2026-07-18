"""Resolve primary project entry head lines from step-12 boundary labels."""

from __future__ import annotations


def resolve_project_entry_heads(tokens: list[dict]) -> set[tuple[int, int]]:
    """Lines with B-PROJ_START from project_phase2_divider (step 12)."""
    heads: set[tuple[int, int]] = set()
    for t in tokens:
        label = t.get("bioLabel") or t.get("bio_label") or "O"
        if label == "B-PROJ_START":
            heads.add((
                int(t.get("page", 0)),
                int(t.get("lineIndex", t.get("line_index", 0))),
            ))
    return heads
