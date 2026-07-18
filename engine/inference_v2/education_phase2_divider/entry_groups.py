"""Group EDUCATION tokens into per-entry blocks using entry-head lines.

Mirrors experience ``group_experience_tokens_by_heads`` and project
``group_project_tokens_by_heads``: entries are sliced purely on ``lineIndex``
head boundaries (table-jitter safe), so the structured view reflects the same
dividers as step 5.
"""

from __future__ import annotations


def group_education_tokens_by_heads(
    tokens: list[dict],
    head_lines: set[tuple[int, int]],
) -> list[list[dict]]:
    """Slice EDUCATION tokens into entries at entry-head lines (line-index based)."""
    if not tokens:
        return []
    if not head_lines:
        return [tokens]

    ordered = sorted(
        tokens,
        key=lambda t: (
            int(t.get("page", 0)),
            int(t.get("lineIndex", t.get("line_index", 0))),
            int(t.get("tokenIndex", t.get("token_index", 0))),
            float(t.get("x0", 0.0)),
        ),
    )

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
