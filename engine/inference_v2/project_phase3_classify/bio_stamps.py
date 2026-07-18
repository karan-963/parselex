"""Stamp resolved macro labels onto segment tokens as BIO tags.

Two continuity fixes on top of the raw per-segment classification, both
gap-filling only (never overriding a segment the model confidently
classified as something else):

1. Bridge — a punctuation-only segment (e.g. a lone "—") sandwiched between
   two segments the model gave the *same* macro label almost always means
   the model mislabeled that connector token instead of continuing the
   surrounding field (e.g. "Resume TLM" / "—" / "Advanced NLP Extraction
   Engine" is one PROJECT_NAME, not PROJECT_NAME / DESC / PROJECT_NAME).
2. Merge — once bridged, adjacent segments sharing the same macro label are
   stitched into one continuous BIO span (B- then I-...) instead of each
   restarting with its own B-, unless the new segment genuinely opens a new
   bulleted list item.
"""

from __future__ import annotations

import re

from .config import MACRO_TO_BIO_PREFIX

_ALNUM_RE = re.compile(r"[a-zA-Z0-9]")
# Real list-item bullets only — a bare dash/em-dash is glue (title separator,
# date-range dash), not a hard "new item" signal, so it's excluded here.
_BULLET_CHARS = frozenset({"•", "◦", "▪", "■", "●", "❖", "·", "*"})


def _seg_text(seg_tokens: list[dict]) -> str:
    return "".join((t.get("token") or "").strip() for t in seg_tokens if t)


def _is_punctuation_only(seg_tokens: list[dict]) -> bool:
    text = _seg_text(seg_tokens)
    return bool(text) and not _ALNUM_RE.search(text)


def _is_bullet_start(seg_tokens: list[dict]) -> bool:
    for tok in seg_tokens:
        text = (tok.get("token") or "").strip() if tok else ""
        if not text:
            continue
        return text in _BULLET_CHARS
    return False


def _bridge_punctuation_gaps(
    project_segments: list[dict],
    macros: list[str],
) -> list[str]:
    result = list(macros)
    for i in range(1, len(project_segments) - 1):
        seg_tokens = project_segments[i].get("tokens") or []
        if not _is_punctuation_only(seg_tokens):
            continue
        prev_macro, next_macro = result[i - 1], result[i + 1]
        if prev_macro and prev_macro == next_macro and prev_macro != result[i]:
            result[i] = prev_macro
    return result


def stamp_segment_tokens(
    seg_tokens: list[dict],
    macro_label: str,
    final_labels: dict[int, str],
    *,
    continue_prev: bool = False,
) -> None:
    prefix = MACRO_TO_BIO_PREFIX.get(macro_label)
    if not prefix:
        return
    for j, tok in enumerate(seg_tokens):
        if j == 0:
            bio = f"I-{prefix}" if continue_prev else f"B-{prefix}"
        else:
            bio = f"I-{prefix}"
        final_labels[id(tok)] = bio


def apply_bio_stamps(
    project_segments: list[dict],
    resolved_labels: list[str],
) -> dict[int, str]:
    macros = _bridge_punctuation_gaps(project_segments, resolved_labels)

    final_labels: dict[int, str] = {}
    prev_macro: str | None = None
    for seg, macro in zip(project_segments, macros):
        seg_tokens = seg.get("tokens") or []
        continue_prev = (
            prev_macro is not None
            and macro == prev_macro
            and not _is_bullet_start(seg_tokens)
        )
        stamp_segment_tokens(seg_tokens, macro, final_labels, continue_prev=continue_prev)
        if macro in MACRO_TO_BIO_PREFIX:
            prev_macro = macro
    return final_labels
