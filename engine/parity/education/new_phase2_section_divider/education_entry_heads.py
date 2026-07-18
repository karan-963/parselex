"""Resolve education entry-head (page, lineIndex) to page-relative token keys."""

from __future__ import annotations


def build_global_line_map(tokens: list[dict]) -> dict[tuple[int, int], tuple[int, int]]:
    """Map ``(page, document_global_line_index)`` → ``(page, page_relative_line_index)``."""
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    current_page: int | None = None
    current_line: int | None = None
    global_counter = -1

    for t in tokens:
        p = int(t.get("page", 0))
        l = int(t.get("lineIndex", t.get("line_index", 0)))
        if p != current_page or l != current_line:
            global_counter += 1
            current_page = p
            current_line = l
            mapping[(p, global_counter)] = (p, l)
    return mapping


def _line_has_tokens(tokens: list[dict], page: int, line_index: int) -> bool:
    return any(
        int(t.get("page", 0)) == page and int(t.get("lineIndex", t.get("line_index", 0))) == line_index
        for t in tokens
    )


def _resolve_one_head(
    tokens: list[dict],
    page: int,
    line_index: int,
    global_map: dict[tuple[int, int], tuple[int, int]],
) -> tuple[int, int]:
    direct = (page, line_index)
    if _line_has_tokens(tokens, page, line_index):
        return direct
    return global_map.get((page, line_index), direct)


def resolve_education_entry_heads(
    tokens: list[dict],
    heads: list[dict],
) -> set[tuple[int, int]]:
    """Resolve labeled heads to page-relative ``(page, lineIndex)`` keys."""
    if not heads:
        return set()
    global_map = build_global_line_map(tokens)
    resolved: set[tuple[int, int]] = set()
    for h in heads:
        if not isinstance(h, dict) or "page" not in h or "lineIndex" not in h:
            continue
        resolved.add(_resolve_one_head(tokens, int(h["page"]), int(h["lineIndex"]), global_map))
    return resolved
