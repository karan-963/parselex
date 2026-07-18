"""Date anchor patterns for experience entry boundary heuristics (step 8)."""

from __future__ import annotations

import re

MONTH_NAMES = frozenset({
    "jan", "january", "feb", "february", "mar", "march", "apr", "april",
    "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
    "september", "oct", "october", "nov", "november", "dec", "december",
})

_APOSTROPHE = r"[''\u2018\u2019\u2032]"
_MONTH_ALT = "|".join(sorted((re.escape(m) for m in MONTH_NAMES), key=len, reverse=True))

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
APOSTROPHE_YEAR_RE = re.compile(rf"{_APOSTROPHE}\s*(?:\d{{4}}|\d{{2}})\b")
MONTH_YEAR_RE = re.compile(
    rf"\b(?:{_MONTH_ALT})\.?\s*(?:{_APOSTROPHE}\s*)?(?:(?:19|20)\d{{2}}|\d{{2}})\b",
    re.I,
)
DATE_END_RE = re.compile(r"\b(present|current|ongoing|now)\b", re.I)
DATE_TOKEN_RE = re.compile(
    rf"(?:{_APOSTROPHE}\s*)?(?:\d{{4}}|\d{{2}})\b|"
    rf"\b(?:{_MONTH_ALT})\.?\s*(?:{_APOSTROPHE}\s*)?(?:(?:19|20)\d{{2}}|\d{{2}})\b|"
    rf"\b(?:{_MONTH_ALT})\b|"
    rf"\b(19|20)\d{{2}}\b|"
    rf"\b(present|current|ongoing|now)\b",
    re.I,
)


def is_date_token(token: str) -> bool:
    """True when a single PDF token is part of a date phrase."""
    t = (token or "").strip()
    if not t:
        return False
    if DATE_END_RE.fullmatch(t):
        return True
    if YEAR_RE.fullmatch(t):
        return True
    if APOSTROPHE_YEAR_RE.search(t) or MONTH_YEAR_RE.search(t):
        return True
    if _is_month_token(t):
        return True
    return False


def _is_month_token(token: str) -> bool:
    return (token or "").strip().lower().rstrip(".") in MONTH_NAMES


def has_date_anchor(text: str) -> bool:
    lower = text.lower()
    if YEAR_RE.search(text) or APOSTROPHE_YEAR_RE.search(text) or MONTH_YEAR_RE.search(text):
        return True
    if DATE_END_RE.search(lower):
        return True
    return any(re.search(rf"\b{re.escape(m)}\b", lower) for m in MONTH_NAMES)


def first_date_anchor_pos(text: str) -> int | None:
    """Return start index of the earliest date anchor in text."""
    positions: list[int] = []
    for pat in (YEAR_RE, APOSTROPHE_YEAR_RE, MONTH_YEAR_RE):
        m = pat.search(text)
        if m:
            positions.append(m.start())
    lower = text.lower()
    for m in MONTH_NAMES:
        hit = re.search(rf"\b{re.escape(m)}\b", lower)
        if hit:
            positions.append(hit.start())
    end_m = DATE_END_RE.search(lower)
    if end_m:
        positions.append(end_m.start())
    return min(positions) if positions else None


def find_date_tokens(tokens: list[dict]) -> list[int]:
    """Return token indices that match date patterns (for diagnostics)."""
    hits: list[int] = []
    for idx, tok in enumerate(tokens):
        if is_date_token(tok.get("token", "")):
            hits.append(idx)
            continue
        if has_date_anchor(tok.get("token", "")):
            hits.append(idx)
    return hits
