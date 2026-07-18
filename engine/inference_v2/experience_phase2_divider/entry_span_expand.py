"""Expand B-ENTRY head lines into full entry spans (B/I-ENTRY until next entry)."""

from __future__ import annotations

from collections import defaultdict


def _line_key(token: dict) -> tuple[int, int]:
    return (int(token.get("page", 0)), int(token.get("lineIndex", token.get("line_index", 0))))


def _has_alnum(text: str) -> bool:
    return any(c.isalnum() for c in text)


def _is_structural_only_line(indices: list[int], tokens: list[dict]) -> bool:
  line_text = "".join((tokens[i].get("token") or "") for i in indices).strip()
  if not line_text:
    return True
  if line_text in {"-", "–", "—", "|"}:
    return True
  return not _has_alnum(line_text)


def expand_entry_span_labels(tokens: list[dict], word_preds: list[str]) -> list[str]:
    """Mark all tokens between entry heads as I-ENTRY; first alnum on each head line = B-ENTRY."""
    if not tokens or len(word_preds) != len(tokens):
        return word_preds

    by_line: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, tok in enumerate(tokens):
        by_line[_line_key(tok)].append(idx)

    ordered_lines = sorted(by_line.keys(), key=lambda k: (k[0], k[1]))
    if not ordered_lines:
        return word_preds

    b_entry_lines: list[tuple[int, int]] = []
    for line_key in ordered_lines:
        indices = sorted(by_line[line_key], key=lambda i: tokens[i].get("tokenIndex", i))
        if any(word_preds[i] == "B-ENTRY" for i in indices):
            b_entry_lines.append(line_key)

    if not b_entry_lines:
        return word_preds

    line_pos = {lk: i for i, lk in enumerate(ordered_lines)}
    span_ranges: list[tuple[int, int]] = []
    for bi, bl in enumerate(b_entry_lines):
        start = line_pos[bl]
        end = line_pos[b_entry_lines[bi + 1]] if bi + 1 < len(b_entry_lines) else len(ordered_lines)
        span_ranges.append((start, end))

    in_span = set()
    for start, end in span_ranges:
        in_span.update(ordered_lines[start:end])

    result = ["O"] * len(tokens)
    for line_key in ordered_lines:
        if line_key not in in_span:
            continue
        indices = sorted(by_line[line_key], key=lambda i: tokens[i].get("tokenIndex", i))
        if _is_structural_only_line(indices, tokens):
            continue
        is_head_line = line_key in b_entry_lines
        placed_b = False
        for idx in indices:
            tok_text = (tokens[idx].get("token") or "").strip()
            if is_head_line and not placed_b and _has_alnum(tok_text):
                result[idx] = "B-ENTRY"
                placed_b = True
            else:
                result[idx] = "I-ENTRY"

    return result
