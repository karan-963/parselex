"""Group EXPERIENCE tokens into job entries using resolved entry-head lines."""

from __future__ import annotations

from .entry_slice_heads import resolve_segmentation_entry_heads


def group_experience_tokens_by_heads(
    tokens: list[dict],
    resume_id: str = "",
    slug: str | None = None,
) -> list[list[dict]]:
    """Slice EXPERIENCE tokens into per-job blocks at entry-head lines.

    Mirrors the entry grouping used in phase-1 segmentation so the structured
    view reflects the same job dividers as step 9. Returns a single block when no
    heads resolve (keeps downstream extraction working).
    """
    exp = [
        t for t in tokens
        if t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]
    if not exp:
        return []

    ordered = sorted(
        exp,
        key=lambda t: (
            t.get("page", 0),
            t.get("lineIndex", t.get("line_index", 0)),
            t.get("tokenIndex", t.get("token_index", 0)),
            t.get("x0", 0),
        ),
    )

    # Prefer step-9 divider heads stamped on tokens (survive phase-1/phase-3 relabel).
    head_lines = {
        (int(t.get("page", 0)), int(t.get("lineIndex", t.get("line_index", 0))))
        for t in ordered
        if t.get("_expEntryHead")
    }
    if not head_lines:
        head_lines = resolve_segmentation_entry_heads(ordered, resume_id, slug)
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
