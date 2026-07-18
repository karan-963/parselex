"""Stamp resolved segment BIO labels onto personal tokens."""

from __future__ import annotations


def _normalize_label(label: str) -> str:
    if not label or label == "O":
        return "O"
    if label.startswith("I-"):
        return f"B-{label[2:]}"
    return label


def stamp_personal_segment(seg_tokens: list[dict], label: str, final_labels: dict[int, str]) -> None:
    norm = _normalize_label(label)
    if norm == "O":
        for tok in seg_tokens:
            if tok and isinstance(tok, dict):
                final_labels[id(tok)] = "O"
        return

    if not norm.startswith("B-"):
        return

    entity = norm[2:]
    for index, tok in enumerate(seg_tokens):
        if tok and isinstance(tok, dict):
            final_labels[id(tok)] = f"B-{entity}" if index == 0 else f"I-{entity}"


def apply_bio_stamps(personal_segments: list[dict], resolved_labels: list[str]) -> dict[int, str]:
    final_labels: dict[int, str] = {}
    for segment, label in zip(personal_segments, resolved_labels):
        stamp_personal_segment(segment.get("tokens") or [], label, final_labels)
    return final_labels
