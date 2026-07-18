"""Post-inference segmentation fixes: column gaps, style shifts, date/entity boundaries."""

from __future__ import annotations

import re
from collections import defaultdict

B_SEG = 1
I_SEG = 2

GAP_MEDIAN_MULTIPLIER = 3.0
GAP_MIN_THRESHOLD_PT = 40.0
MODERATE_GAP_PT = 20.0
STYLE_GAP_PT = 35.0
FONT_SIZE_SHIFT_PT = 0.5

DATE_END_KEYWORDS = frozenset({"present", "current", "ongoing", "now", "till"})
DATE_DELIMITERS = frozenset({"-", "–", "—", "to"})
MONTH_NAMES = frozenset({
    "jan", "january", "feb", "february", "mar", "march", "apr", "april",
    "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
    "september", "oct", "october", "nov", "november", "dec", "december",
})
EDUCATION_BIO_BOUNDARIES = frozenset({
    "DEG", "INST", "SDATE", "EDATE", "GPA", "LOC", "DESC",
})
EDUCATION_HEADER_ENTITIES = frozenset({
    "DEG", "INST", "GPA", "LOC", "DESC",
})
BULLET_GLYPHS = frozenset({
    "•", "●", "❖", "▪", "◦", "■", "✓", "✔", "·", "*", "-", "–", "—",
    "\uf0b7", "\uf0a7",
})
YEAR_RE = re.compile(r"^(19|20)\d{2}$")
COMPACT_YEAR_RE = re.compile(r"^-?('?\d{2}|(19|20)\d{2})$")


def _token_bio(token: dict) -> str:
    """Prefer Mongo overlay used during inference; fall back to bioLabel."""
    return token.get("_fieldBioLabel") or token.get("bioLabel", "O")


def _is_text_token(token: str) -> bool:
    return bool(re.search(r"[a-zA-Z0-9]", token or ""))


def _entity_type(bio: str) -> str:
    if bio.startswith("B-") or bio.startswith("I-"):
        return bio[2:]
    return bio


def _is_year_token(token: str) -> bool:
    return bool(YEAR_RE.match((token or "").strip()))


def _is_month_token(token: str) -> bool:
    return (token or "").strip().lower().rstrip(".") in MONTH_NAMES


def _is_date_opener_token(token: str) -> bool:
    text = (token or "").strip()
    if not text:
        return False
    if _is_year_token(text) or _is_month_token(text):
        return True
    return bool(COMPACT_YEAR_RE.match(text))


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
        bio = _token_bio(entry_tokens[j])
        if _is_year_token(tok) or _is_month_token(tok) or _entity_type(bio) in ("SDATE", "EDATE"):
            return True
        if _is_skippable_token(tok):
            j = _walk_prev_meaningful(entry_tokens, j)
            continue
        break
    return False


def _follows_completed_date(entry_tokens: list[dict], idx: int) -> bool:
    j = _walk_prev_meaningful(entry_tokens, idx)
    if j is None:
        return False

    prev = entry_tokens[j]
    prev_tok = (prev.get("token") or "").strip()
    prev_bio = _token_bio(prev)
    prev_type = _entity_type(prev_bio)

    if prev_type in ("SDATE", "EDATE") or _is_year_token(prev_tok):
        return True
    if prev_tok in DATE_DELIMITERS:
        j2 = _walk_prev_meaningful(entry_tokens, j)
        if j2 is not None:
            t2 = (entry_tokens[j2].get("token") or "").strip()
            b2 = _token_bio(entry_tokens[j2])
            if _is_year_token(t2) or _entity_type(b2) in ("SDATE", "EDATE"):
                return True
    return False


def _starts_header_field(token: dict, *, use_bio_hints: bool) -> bool:
    bio = _token_bio(token)
    if use_bio_hints and bio.startswith("B-") and _entity_type(bio) in EDUCATION_HEADER_ENTITIES:
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


def apply_style_boundary_heuristic(
    entry_tokens: list[dict],
    pred_labels: list[int],
    *,
    skip_indices: set[int] | None = None,
) -> list[int]:
    """Split phrases when font style or moderate horizontal gap shifts (project P1/P3 rule)."""
    if not entry_tokens or len(pred_labels) != len(entry_tokens):
        return pred_labels

    skip = skip_indices or set()
    result = list(pred_labels)

    by_line: dict[tuple, list[int]] = defaultdict(list)
    for idx, tok in enumerate(entry_tokens):
        by_line[(tok.get("page"), tok.get("lineIndex"))].append(idx)

    for line_indices in by_line.values():
        for i in range(1, len(line_indices)):
            curr_i = line_indices[i]
            if curr_i in skip:
                continue
            prev_i = line_indices[i - 1]
            prev_t = entry_tokens[prev_i]
            curr_t = entry_tokens[curr_i]
            if not (_is_text_token(prev_t.get("token", "")) and _is_text_token(curr_t.get("token", ""))):
                continue

            prev_text = (prev_t.get("token") or "").strip()
            curr_text = (curr_t.get("token") or "").strip()
            if prev_text in BULLET_GLYPHS or curr_text in BULLET_GLYPHS:
                continue

            bold_shift = bool(prev_t.get("isBold")) != bool(curr_t.get("isBold"))
            size_shift = abs(float(prev_t.get("fontSize", 9.0)) - float(curr_t.get("fontSize", 9.0))) > FONT_SIZE_SHIFT_PT
            word_gap = _horizontal_gap(prev_t, curr_t) > STYLE_GAP_PT
            if bold_shift or size_shift or word_gap:
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

        bio = _token_bio(tok)
        entity = _entity_type(bio)

        if use_bio_hints and bio.startswith("B-") and entity in EDUCATION_BIO_BOUNDARIES:
            result[idx] = B_SEG
            continue

        if not use_bio_hints and _is_date_opener_token(tok.get("token", "")):
            if not _is_date_range_tail(entry_tokens, idx):
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
    result = apply_style_boundary_heuristic(entry_tokens, result, skip_indices=skip_indices)
    result = apply_entity_and_date_heuristics(
        entry_tokens, result, skip_indices=skip_indices, use_bio_hints=use_bio_hints
    )
    return result
