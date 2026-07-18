"""Style-only post-process layer on top of model boundary predictions."""

from __future__ import annotations


def merge_promote_only(
    model_lines: set[tuple[int, int]],
    extra_lines: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Keep every model head; add style-promoted lines only."""
    return set(model_lines) | set(extra_lines)
