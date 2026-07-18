"""Post-inference segmentation fixes: column gaps, date phrases, entity boundaries."""

from __future__ import annotations

import re
from collections import defaultdict

B_SEG = 1
I_SEG = 2

GAP_MEDIAN_MULTIPLIER = 3.0
GAP_MIN_THRESHOLD_PT = 40.0
MODERATE_GAP_PT = 20.0

DATE_END_KEYWORDS = frozenset({"present", "current", "ongoing", "now", "till"})
DATE_DELIMITERS = frozenset({"-", "–", "—", "to"})
BULLET_CHARS = frozenset({"•", "◦", "▪", "■", "●", "❖", "∗", "*", "·", "\uf0b7", "\uf0a7"})
MONTH_NAMES = frozenset({
    "jan", "january", "feb", "february", "mar", "march", "apr", "april",
    "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
    "september", "oct", "october", "nov", "november", "dec", "december",
})
HEADER_ENTITY_TYPES = frozenset({"ROLE", "COMP", "COMP_LOC", "SDATE", "EDATE", "ROLEMETA"})
BOUNDARY_LABEL_TYPES = frozenset({"ENTRY", "SEG"})
YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_APOSTROPHE = r"[''\u2018\u2019\u2032]"
APOSTROPHE_YEAR_RE = re.compile(rf"{_APOSTROPHE}\s*(?:\d{{4}}|\d{{2}})\b")
_MONTH_ALT = "|".join(sorted((re.escape(m) for m in MONTH_NAMES), key=len, reverse=True))
MONTH_YEAR_RE = re.compile(
    rf"\b(?:{_MONTH_ALT})\.?\s*(?:{_APOSTROPHE}\s*)?(?:(?:19|20)\d{{2}}|\d{{2}})\b",
    re.I,
)


def _is_text_token(token: str) -> bool:
    return bool(re.search(r"[a-zA-Z0-9]", token or ""))


def _field_bio(token: dict) -> str:
    return token.get("_fieldBioLabel") or token.get("bioLabel", "O")


def _entity_type(bio: str) -> str:
    if bio.startswith("B-") or bio.startswith("I-"):
        return bio[2:]
    return bio


def _is_year_token(token: str) -> bool:
    t = (token or "").strip()
    if YEAR_RE.fullmatch(t):
        return True
    return bool(APOSTROPHE_YEAR_RE.search(t))


def _is_month_token(token: str) -> bool:
    t = (token or "").strip()
    if t.lower().rstrip(".") in MONTH_NAMES:
        return True
    return bool(MONTH_YEAR_RE.search(t))


def _same_line(a: dict, b: dict) -> bool:
    return a.get("page") == b.get("page") and a.get("lineIndex") == b.get("lineIndex")


def _horizontal_gap(prev_t: dict, curr_t: dict) -> float:
    return curr_t.get("x0", 0.0) - prev_t.get("x1", 0.0)


def compute_line_gap_threshold(gaps: list[float]) -> float:
    if not gaps:
        return GAP_MIN_THRESHOLD_PT
    med = sorted(gaps)[len(gaps) // 2]
    return max(GAP_MEDIAN_MULTIPLIER * med, GAP_MIN_THRESHOLD_PT)


def _is_skippable_token(token: str) -> bool:
    t = (token or "").strip()
    return not t or t in {'"', "'", "‘", "’", ",", "•", "●", "▪"}


def _walk_prev_meaningful(entry_tokens: list[dict], idx: int) -> int | None:
    j = idx - 1
    while j >= 0:
        if not _same_line(entry_tokens[j], entry_tokens[idx]):
            return None
        tok = (entry_tokens[j].get("token") or "").strip()
        if _is_skippable_token(tok):
            j -= 1
            continue
        return j
    return None


def _is_date_range_tail(entry_tokens: list[dict], idx: int) -> bool:
    """True when token continues 'Month Year - Present' on one line."""
    token = (entry_tokens[idx].get("token") or "").strip().lower()
    if token not in DATE_END_KEYWORDS:
        return False

    j = _walk_prev_meaningful(entry_tokens, idx)
    if j is None:
        return False

    delim = (entry_tokens[j].get("token") or "").strip()
    if delim not in DATE_DELIMITERS:
        return False
    if _horizontal_gap(entry_tokens[j], entry_tokens[idx]) > 8.0:
        return False

    j = _walk_prev_meaningful(entry_tokens, j)
    while j is not None:
        tok = (entry_tokens[j].get("token") or "").strip()
        bio = entry_tokens[j].get("bioLabel", "O")
        if _is_year_token(tok) or _is_month_token(tok) or _entity_type(bio) in ("SDATE", "EDATE"):
            return True
        if _is_skippable_token(tok):
            j = _walk_prev_meaningful(entry_tokens, j)
            continue
        break
    return False


def _follows_completed_date(entry_tokens: list[dict], idx: int) -> bool:
    """True when idx starts a new header field immediately after a date/year."""
    j = _walk_prev_meaningful(entry_tokens, idx)
    if j is None:
        return False

    prev = entry_tokens[j]
    prev_tok = (prev.get("token") or "").strip()
    prev_bio = prev.get("bioLabel", "O")
    prev_type = _entity_type(prev_bio)

    if prev_type in ("SDATE", "EDATE") or _is_year_token(prev_tok):
        return True
    if prev_tok in DATE_DELIMITERS:
        j2 = _walk_prev_meaningful(entry_tokens, j)
        if j2 is not None:
            t2 = (entry_tokens[j2].get("token") or "").strip()
            b2 = entry_tokens[j2].get("bioLabel", "O")
            if _is_year_token(t2) or _entity_type(b2) in ("SDATE", "EDATE"):
                return True
    return False


def _starts_header_field(token: dict, *, use_bio_hints: bool) -> bool:
    bio = _field_bio(token)
    if use_bio_hints and bio.startswith("B-") and _entity_type(bio) in {
        "ROLE", "COMP", "COMP_LOC", "SDATE", "EDATE", "ROLEMETA",
    }:
        return True
    text = (token.get("token") or "").strip()
    if not _is_text_token(text):
        return False
    if text.lower() in DATE_END_KEYWORDS or _is_year_token(text) or _is_month_token(text):
        return False
    return True


def apply_gap_boundary_heuristic(
    entry_tokens: list[dict],
    pred_labels: list[int],
    *,
    skip_indices: set[int] | None = None,
) -> list[int]:
    if not entry_tokens or len(pred_labels) != len(entry_tokens):
        return pred_labels

    skip = skip_indices or set()
    result = list(pred_labels)

    by_line: dict[tuple, list[int]] = defaultdict(list)
    for idx, tok in enumerate(entry_tokens):
        by_line[(tok.get("page"), tok.get("lineIndex"))].append(idx)

    for line_indices in by_line.values():
        if len(line_indices) < 2:
            continue

        gaps = [
            _horizontal_gap(entry_tokens[line_indices[i - 1]], entry_tokens[line_indices[i]])
            for i in range(1, len(line_indices))
        ]
        threshold = compute_line_gap_threshold(gaps)

        for i in range(1, len(line_indices)):
            curr_i = line_indices[i]
            if curr_i in skip:
                continue
            prev_i = line_indices[i - 1]
            prev_t = entry_tokens[prev_i]
            curr_t = entry_tokens[curr_i]
            if not (_is_text_token(prev_t.get("token", "")) and _is_text_token(curr_t.get("token", ""))):
                continue
            if _horizontal_gap(prev_t, curr_t) >= threshold:
                result[curr_i] = B_SEG

    return result


def apply_entity_and_date_heuristics(
    entry_tokens: list[dict],
    pred_labels: list[int],
    *,
    skip_indices: set[int] | None = None,
    use_bio_hints: bool = True,
) -> list[int]:
    if not entry_tokens or len(pred_labels) != len(entry_tokens):
        return pred_labels

    skip = skip_indices or set()
    result = list(pred_labels)

    for idx, tok in enumerate(entry_tokens):
        if idx in skip:
            continue

        bio = _field_bio(tok)
        entity = _entity_type(bio)
        tok_str = (tok.get("token") or "").strip()

        if tok_str in BULLET_CHARS:
            continue
        if not _is_text_token(tok_str):
            continue

        # Only promote header entity fields — not B-DESC, B-ENTRY boundary labels, etc.
        if use_bio_hints and entity in HEADER_ENTITY_TYPES and bio.startswith("B-"):
            result[idx] = B_SEG
            continue

        if _is_date_range_tail(entry_tokens, idx):
            result[idx] = I_SEG
            continue

        if not _starts_header_field(tok, use_bio_hints=use_bio_hints):
            continue

        prev_i = _walk_prev_meaningful(entry_tokens, idx)
        if prev_i is None:
            continue

        if not _follows_completed_date(entry_tokens, idx):
            continue

        gap = _horizontal_gap(entry_tokens[prev_i], tok)
        if gap >= MODERATE_GAP_PT:
            result[idx] = B_SEG

    return result


def apply_segment_postprocess(
    entry_tokens: list[dict],
    pred_labels: list[int],
    *,
    skip_indices: set[int] | None = None,
    use_bio_hints: bool = True,
) -> list[int]:
    result = apply_gap_boundary_heuristic(entry_tokens, pred_labels, skip_indices=skip_indices)
    result = apply_entity_and_date_heuristics(
        entry_tokens, result, skip_indices=skip_indices, use_bio_hints=use_bio_hints
    )
    return result
