"""Group PROJECT tokens into per-entry blocks using step-12 divider heads."""

from __future__ import annotations

from .entry_slice_heads import resolve_project_entry_heads
from .heads_loader import load_entry_head_lines


def group_project_tokens_by_heads(
    tokens: list[dict],
    resume_id: str = "",
) -> list[list[dict]]:
    """Slice PROJECT tokens into per-project blocks at entry-head lines.

    Mirrors experience ``group_experience_tokens_by_heads`` so the structured
    view reflects the same entry dividers as step 12.
    """
    proj = [
        t for t in tokens
        if t.get("section") in ("PROJECT", "PROJECTS")
        and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]
    if not proj:
        return []

    ordered = sorted(
        proj,
        key=lambda t: (
            t.get("page", 0),
            t.get("lineIndex", t.get("line_index", 0)),
            t.get("tokenIndex", t.get("token_index", 0)),
            t.get("x0", 0),
        ),
    )

    head_lines = {
        (int(t.get("page", 0)), int(t.get("lineIndex", t.get("line_index", 0))))
        for t in ordered
        if t.get("_projEntryHead")
    }
    if not head_lines:
        mongo_heads = load_entry_head_lines(resume_id) if resume_id else set()
        if mongo_heads:
            proj_line_keys = {
                (int(t.get("page", 0)), int(t.get("lineIndex", t.get("line_index", 0))))
                for t in ordered
            }
            head_lines = mongo_heads & proj_line_keys
    if not head_lines:
        head_lines = resolve_project_entry_heads(ordered)
    if not head_lines:
        return [ordered]

    entries: list[list[dict]] = []
    current_block: list[dict] = []
    current_head: tuple[int, int] | None = None

    for t in ordered:
        key = (int(t.get("page", 0)), int(t.get("lineIndex", t.get("line_index", 0))))
        if key in head_lines and key != current_head:
            if current_block:
                entries.append(current_block)
                current_block = []
            current_head = key
        if current_head is not None:
            current_block.append(t)

    if current_block:
        entries.append(current_block)

    return entries or [ordered]
