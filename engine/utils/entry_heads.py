"""Resolve entry-head (page, lineIndex) markers to page-relative token keys."""

from __future__ import annotations

import re


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


def _line_words(tokens: list[dict], page: int, line_index: int) -> list[str]:
    toks = [
        t for t in tokens
        if int(t.get("page", 0)) == page and int(t.get("lineIndex", t.get("line_index", 0))) == line_index
    ]
    toks.sort(key=lambda t: t.get("tokenIndex", t.get("token_index", 0)))
    text = " ".join((t.get("token") or "").strip() for t in toks)
    return [w for w in re.split(r"\s+", text.strip()) if w]


def _word_match(a: str, b: str) -> bool:
    al, bl = a.lower(), b.lower()
    return al == bl or al in bl or bl in al


def _match_head_by_content(
    tokens: list[dict],
    page: int,
    reference_words: list[str],
) -> tuple[int, int] | None:
    """Find inference line on ``page`` whose leading tokens match Mongo head text."""
    if not reference_words:
        return None

    lines: dict[tuple[int, int], list[dict]] = {}
    for t in tokens:
        if int(t.get("page", 0)) != page:
            continue
        key = (page, int(t.get("lineIndex", t.get("line_index", 0))))
        lines.setdefault(key, []).append(t)

    best_key: tuple[int, int] | None = None
    best_score = 0.0
    min_prefix = min(3, len(reference_words))

    for key, line_toks in lines.items():
        words = _line_words(tokens, key[0], key[1])
        if not words:
            continue

        prefix = 0
        for ref, got in zip(reference_words, words):
            if _word_match(ref, got):
                prefix += 1
            else:
                break

        overlap = len({w.lower() for w in reference_words} & {w.lower() for w in words})
        ratio = overlap / max(len(reference_words), 1)
        score = prefix * 2.0 + ratio
        if prefix >= min_prefix and score > best_score:
            best_score = score
            best_key = key

    return best_key


def _resolve_one_head(
    tokens: list[dict],
    page: int,
    line_index: int,
    global_map: dict[tuple[int, int], tuple[int, int]],
    reference_tokens: list[dict] | None = None,
) -> tuple[int, int]:
    """Prefer page-relative ``(page, lineIndex)`` when tokens exist on that line."""
    if reference_tokens:
        ref_words = _line_words(reference_tokens, page, line_index)
        matched = _match_head_by_content(tokens, page, ref_words)
        if matched is not None:
            return matched

    direct = (page, line_index)
    if _line_has_tokens(tokens, page, line_index):
        return direct
    return global_map.get((page, line_index), direct)


def resolve_entry_head_lines(
    tokens: list[dict],
    heads: list[dict],
    reference_tokens: list[dict] | None = None,
) -> set[tuple[int, int]]:
    """
    Resolve entry heads to page-relative ``(page, lineIndex)`` keys.

    Labeling app stores page-relative ``lineIndex``. Older exports may use
    document-global indices — those are mapped via ``build_global_line_map``
    when no tokens exist on the direct page-relative line.
    """
    if not heads:
        return set()

    global_map = build_global_line_map(tokens)
    resolved: set[tuple[int, int]] = set()
    for h in heads:
        resolved.add(
            _resolve_one_head(
                tokens,
                int(h["page"]),
                int(h["lineIndex"]),
                global_map,
                reference_tokens,
            )
        )
    return resolved


def resolve_entry_head_list(
    tokens: list[dict],
    heads: list[dict],
    reference_tokens: list[dict] | None = None,
) -> list[tuple[int, int]]:
    """Sorted page-relative head keys, preserving head order where possible."""
    order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    global_map = build_global_line_map(tokens)
    for h in heads:
        hp = int(h["page"])
        hl = int(h["lineIndex"])
        key = _resolve_one_head(tokens, hp, hl, global_map, reference_tokens)
        if key not in seen:
            seen.add(key)
            order.append(key)
    return sorted(order)
