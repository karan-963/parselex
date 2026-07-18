"""Personal-section heuristic segmentation — atomic field segments for classification."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from strategy import _is_phone_token

_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
_HARD_SEPARATORS = {"|", "—", "–", "•", "·", "▪", "■", "/", ";"}
_GAP_THRESHOLD = 35.0
_HETEROGENEOUS = "HETEROGENEOUS"


def _token_text(tok: dict) -> str:
    return (tok.get("token") or "").strip()


def _bio_entity(lbl: str | None) -> str | None:
    if not lbl or lbl == "O" or "-" not in lbl:
        return None
    return lbl.split("-", 1)[1]


def _field_kind(text: str) -> str | None:
    low = text.lower()
    if _EMAIL_RE.search(low):
        return "EMAIL"
    if "linkedin.com" in low:
        return "LINKEDIN"
    if "github.com" in low:
        return "GITHUB"
    if _is_phone_token(text):
        return "PHONE"
    return None


def _line_field_kind(tokens: list[dict]) -> str | None:
    joined = " ".join(_token_text(t) for t in tokens)
    return _field_kind(joined)


def _is_hard_separator(tok: dict) -> bool:
    return _token_text(tok) in _HARD_SEPARATORS


def _strip_hard_separators(text: str) -> str:
    """Trim hard-separator characters glued to a token's edges (e.g. '|HP')."""
    return text.strip("".join(_HARD_SEPARATORS)).strip()


def _leading_hard_separator(text: str) -> bool:
    return bool(text) and text[0] in _HARD_SEPARATORS


def _trailing_hard_separator(text: str) -> bool:
    return bool(text) and text[-1] in _HARD_SEPARATORS


def _should_split(prev: dict, curr: dict, line_context: str) -> bool:
    if _is_hard_separator(curr) or _is_hard_separator(prev):
        return True

    gap = curr.get("x0", 0.0) - prev.get("x1", 0.0)
    if gap > _GAP_THRESHOLD:
        return True

    prev_lbl = prev.get("bioLabel")
    curr_lbl = curr.get("bioLabel")
    if curr_lbl and curr_lbl.startswith("B-"):
        prev_ent = _bio_entity(prev_lbl)
        curr_ent = _bio_entity(curr_lbl)
        if curr_ent and curr_ent != prev_ent:
            return True

    prev_kind = _field_kind(_token_text(prev)) or _field_kind(line_context)
    curr_kind = _field_kind(_token_text(curr))
    if curr_kind and prev_kind and curr_kind != prev_kind:
        # Avoid splitting http : // fragments mid-URL
        if curr_kind in ("LINKEDIN", "GITHUB") and "http" in line_context.lower():
            if not re.search(r"(linkedin|github)\.com", line_context.lower()):
                return True
        elif curr_kind != prev_kind:
            return True

    return False


def _split_line_tokens(line_tokens: list[dict]) -> list[list[dict]]:
    if not line_tokens:
        return []

    ordered = sorted(line_tokens, key=lambda t: t.get("x0", 0.0))
    chunks: list[list[dict]] = [[]]
    line_context = " ".join(_token_text(t) for t in ordered)

    for tok in ordered:
        text = _token_text(tok)
        if _is_hard_separator(tok):
            if chunks[-1]:
                chunks.append([])
            continue

        # A separator glued to the front of a token (e.g. "|HP") still marks a
        # field boundary, so start a fresh chunk before adding the token.
        starts_new = _leading_hard_separator(text)
        if not chunks[-1]:
            chunks[-1].append(tok)
        else:
            prev = chunks[-1][-1]
            if starts_new or _should_split(prev, tok, line_context):
                chunks.append([tok])
            else:
                chunks[-1].append(tok)

        # A trailing separator (e.g. "email|") closes the current field.
        if _trailing_hard_separator(text):
            chunks.append([])

    return [c for c in chunks if c]


def _tokens_to_segment(tokens: list[dict]) -> dict[str, Any]:
    text = " ".join(
        stripped for t in tokens if (stripped := _strip_hard_separators(_token_text(t)))
    ).strip()
    return {
        "text": text,
        "tokens": tokens,
        "page": tokens[0].get("page", 0),
        "y0": min(t.get("y0", 0.0) for t in tokens),
        "x0": min(t.get("x0", 0.0) for t in tokens),
        "x1": max(t.get("x1", 0.0) for t in tokens),
        "y1": max(t.get("y1", 0.0) for t in tokens),
    }


def build_personal_segments(cleaned_tokens: list[dict]) -> list[dict]:
    """Build atomic personal segments from a cleaned token stream."""
    personal = [
        t for t in cleaned_tokens
        if t.get("section") == "PERSONAL" and _token_text(t)
    ]
    if not personal:
        return []

    personal.sort(key=lambda t: (t.get("page", 0), t.get("lineIndex", 0), t.get("x0", 0.0)))

    line_map: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for t in personal:
        line_map[(t.get("page", 0), t.get("lineIndex", 0))].append(t)

    segments: list[dict] = []
    for key in sorted(line_map.keys()):
        for chunk in _split_line_tokens(line_map[key]):
            seg = _tokens_to_segment(chunk)
            if seg["text"]:
                segments.append(seg)
    return segments


def derive_segment_label(segment: dict, label_list: list[str] | None = None) -> str:
    """Derive segment class from token bioLabels (first-B rule, heterogeneity check)."""
    labels = [t.get("bioLabel") for t in segment.get("tokens", []) if t.get("bioLabel")]
    if not labels:
        return "O"

    b_entities: list[str] = []
    first_b: str | None = None
    for lbl in labels:
        if lbl.startswith("B-"):
            ent = _bio_entity(lbl)
            if ent:
                b_entities.append(ent)
            if first_b is None and lbl in (label_list or []):
                first_b = lbl
            elif first_b is None:
                first_b = lbl

    unique_b = set(b_entities)
    if len(unique_b) > 1:
        return _HETEROGENEOUS

    if first_b:
        if label_list and first_b not in label_list:
            return "O"
        return first_b

    # All I-* or O
    non_o = [l for l in labels if l != "O"]
    if not non_o:
        return "O"
    if all(l.startswith("I-") for l in non_o):
        ent = _bio_entity(non_o[0])
        coerced = f"B-{ent}" if ent else "O"
        if label_list and coerced not in label_list:
            return "O"
        return coerced
    return "O"


def count_b_entities(segment: dict) -> int:
    """Count distinct B- entity types in segment tokens."""
    ents = set()
    for t in segment.get("tokens", []):
        lbl = t.get("bioLabel")
        if lbl and lbl.startswith("B-"):
            ent = _bio_entity(lbl)
            if ent:
                ents.add(ent)
    return len(ents)


def is_mega_segment(segment: dict) -> bool:
    return count_b_entities(segment) > 1


def is_homogeneous_segment(segment: dict) -> bool:
    ents: set[str] = set()
    for t in segment.get("tokens", []):
        lbl = t.get("bioLabel")
        if not lbl or lbl == "O":
            continue
        ent = _bio_entity(lbl)
        if ent:
            ents.add(ent)
    return len(ents) <= 1
