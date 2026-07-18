"""Stamp resolved macro labels onto segment tokens as BIO tags."""

from __future__ import annotations

from .config import MACRO_TO_BIO_PREFIX


def stamp_segment_tokens(seg_tokens: list[dict], macro_label: str, final_labels: dict[int, str]) -> None:
    if macro_label == "DESCRIPTION":
        for tok in seg_tokens:
            if tok and isinstance(tok, dict):
                final_labels[id(tok)] = "O"
        return

    prefix = MACRO_TO_BIO_PREFIX.get(macro_label)
    if not prefix:
        return
    for j, tok in enumerate(seg_tokens):
        if tok and isinstance(tok, dict):
            final_labels[id(tok)] = f"B-{prefix}" if j == 0 else f"I-{prefix}"


def apply_bio_stamps(
    education_segments: list[dict],
    resolved_labels: list[str],
) -> dict[int, str]:
    final_labels: dict[int, str] = {}
    for seg, macro in zip(education_segments, resolved_labels):
        stamp_segment_tokens(seg.get("tokens") or [], macro, final_labels)
    return final_labels
