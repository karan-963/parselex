"""Resolve project entry head lines from boundary predictions."""

from __future__ import annotations


def resolve_project_entry_heads(tokens: list[dict]) -> set[tuple[int, int]]:
    """Lines with B-PROJ_START on any token (entry head line)."""
    heads: set[tuple[int, int]] = set()
    for t in tokens:
        label = t.get("bioLabel") or t.get("bio_label") or t.get("prediction") or "O"
        if label == "B-PROJ_START":
            heads.add((
                int(t.get("page", 0)),
                int(t.get("lineIndex", t.get("line_index", 0))),
            ))
    return heads
