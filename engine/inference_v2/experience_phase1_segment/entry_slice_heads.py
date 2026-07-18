"""Resolve which B-ENTRY lines start a new job entry block for phrase segmentation."""

from __future__ import annotations

from collections import defaultdict

from inference_v2.experience_phase2_divider.date_patterns import has_date_anchor, is_date_token
from inference_v2.experience_phase2_divider.entry_postprocess import (
    _first_meaningful_token,
    _line_text,
    is_bullet_line,
)

MAIN_ENTRY_BULLETS = frozenset({"•", "●", "▪"})
SUB_ENTRY_BULLETS = frozenset({"◦", "∗", "*", "·", "\uf0b7", "\uf0a7"})
STRUCTURAL_PREFIX = frozenset({'"', "'", ",", "|", "(", ")"})


def _line_preds(tokens: list[dict]) -> list[str]:
    return [t.get("bioLabel") or t.get("bio_label") or "O" for t in tokens]


def _first_meaningful_pred(line_toks: list[dict]) -> str | None:
    for t, pred in zip(line_toks, _line_preds(line_toks)):
        tok = (t.get("token") or "").strip()
        if tok and tok not in STRUCTURAL_PREFIX:
            return pred
    return None


def _is_date_only_entry_line(line_toks: list[dict]) -> bool:
    text = _line_text(line_toks)
    first = _first_meaningful_token(line_toks)
    if not first:
        return False
    if first in {"(", "|"} and has_date_anchor(text):
        return True
    if has_date_anchor(text) and not any(
        len((t.get("token") or "").strip()) > 3
        and not is_date_token(t.get("token", ""))
        and (t.get("token") or "").strip() not in SUB_ENTRY_BULLETS | MAIN_ENTRY_BULLETS
        for t in line_toks
    ):
        return True
    return False


def _is_sub_bullet_line(line_toks: list[dict]) -> bool:
    first = _first_meaningful_token(line_toks)
    return first in SUB_ENTRY_BULLETS


def _line_has_b_entry(line_toks: list[dict]) -> bool:
    return any(pred == "B-ENTRY" for pred in _line_preds(line_toks))


def _is_company_or_date_continuation(line_toks: list[dict]) -> bool:
    """Company suffix or parenthetical date line under a job title (not a new entry)."""
    first = _first_meaningful_token(line_toks)
    if not first or first in MAIN_ENTRY_BULLETS:
        return False
    if first in {"(", "|"}:
        return True
    text = _line_text(line_toks).lower()
    if has_date_anchor(_line_text(line_toks)):
        return True
    if any(tok in text for tok in ("pvt", "ltd", "inc", "llc", "limited")):
        return True
    return False


def _is_primary_title_line(line_toks: list[dict]) -> bool:
    first = _first_meaningful_token(line_toks)
    if not first:
        return False
    if first in MAIN_ENTRY_BULLETS:
        return True
    text = _line_text(line_toks)
    if is_bullet_line(line_toks) and not _is_date_only_entry_line(line_toks):
        return True
    if has_date_anchor(text) and len(text.split()) >= 4:
        return True
    return False


def _is_section_heading_line(line_toks: list[dict]) -> bool:
    """EXPERIENCE section title line — not a job entry head for segmentation."""
    text = _line_text(line_toks).strip().lower()
    return text in {"experience", "work experience", "professional experience", "employment"}


def resolve_segmentation_entry_heads(tokens: list[dict], resume_id: str = "", slug: str | None = None) -> set[tuple[int, int]]:
    """Entry slice heads aligned with training evaluate.py (experienceEntryHeads first)."""
    from inference_v2.experience_phase2_divider.heads_loader import load_entry_head_lines

    mongo_heads = load_entry_head_lines(resume_id, tokens, slug) if resume_id else set()
    if mongo_heads:
        return mongo_heads
    return resolve_entry_slice_heads(tokens)


def resolve_entry_slice_heads(tokens: list[dict]) -> set[tuple[int, int]]:
    """Return line keys that begin a new experience entry for step-9 segmentation.

    Uses only lines whose first meaningful token is B-ENTRY, excluding date-only
    continuations and sub-project bullets (◦ Project …).
    """
    by_line: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for t in tokens:
        key = (int(t.get("page", 0)), int(t.get("lineIndex", t.get("line_index", 0))))
        by_line[key].append(t)

    heads: set[tuple[int, int]] = set()
    for line_key, line_toks in by_line.items():
        line_toks.sort(key=lambda t: t.get("tokenIndex", t.get("token_index", 0)))
        if not _line_has_b_entry(line_toks):
            continue
        if _is_date_only_entry_line(line_toks):
            continue
        if _is_sub_bullet_line(line_toks):
            continue
        if _is_company_or_date_continuation(line_toks):
            continue
        if _is_section_heading_line(line_toks):
            continue
        first = _first_meaningful_token(line_toks)
        if first in MAIN_ENTRY_BULLETS and _first_meaningful_pred(line_toks) == "B-ENTRY":
            heads.add(line_key)
            continue
        if _first_meaningful_pred(line_toks) == "B-ENTRY":
            heads.add(line_key)
    return heads
