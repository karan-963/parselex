"""Entry grouping and post-inference guards for education phase 1."""

from __future__ import annotations

import re

from inference_v2.education_phase2_divider.y0_entry_groups import group_education_tokens_by_y0


def group_entries(
    filtered_tokens: list[dict],
    head_lines: set[tuple[int, int]],
) -> list[tuple[tuple[int, int] | None, list[dict]]]:
    """Group tokens into education entries using y0 table rows (not lineIndex walk order)."""
    if not head_lines:
        return [(None, filtered_tokens)]

    y0_groups = group_education_tokens_by_y0(filtered_tokens, head_lines)
    entries: list[tuple[tuple[int, int] | None, list[dict]]] = []
    for block in y0_groups:
        head: tuple[int, int] | None = None
        for token in block:
            key = (int(token["page"]), int(token.get("lineIndex", 0)))
            if key in head_lines or token.get("_eduEntryHead"):
                head = key
                break
        entries.append((head, block))
    return entries


def description_lock_flags(entry_toks: list[dict]) -> list[bool]:
    flags = [False] * len(entry_toks)
    first_desc = -1
    last_desc = -1
    for idx, t in enumerate(entry_toks):
        bio = t.get("_fieldBioLabel") or t.get("bioLabel") or "O"
        if bio in ("B-DESC", "I-DESC"):
            if first_desc == -1:
                first_desc = idx
            last_desc = idx
    if first_desc != -1:
        for idx in range(first_desc, last_desc + 1):
            flags[idx] = True
    return flags


def apply_continuity_guard(entry_toks: list[dict], preds: list[int]) -> list[int]:
    out = list(preds)
    for idx in range(1, len(entry_toks)):
        t_prev = entry_toks[idx - 1]
        t_curr = entry_toks[idx]
        same_line = (
            t_curr.get("page") == t_prev.get("page")
            and t_curr.get("lineIndex") == t_prev.get("lineIndex")
        )
        is_text = bool(re.search(r"[a-zA-Z0-9]", t_curr.get("token", ""))) and bool(
            re.search(r"[a-zA-Z0-9]", t_prev.get("token", ""))
        )
        if same_line and is_text and out[idx] == 1 and out[idx - 1] in (1, 2):
            out[idx] = 2
    return out
