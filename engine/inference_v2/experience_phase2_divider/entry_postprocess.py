"""Experience phase 2 — post-inference B-ENTRY corrections (FP suppress + FN promote)."""

from __future__ import annotations

from collections import defaultdict

from .date_patterns import (
    MONTH_NAMES,
    first_date_anchor_pos,
    has_date_anchor,
    is_date_token,
)

BULLET_CHARS = frozenset({"•", "◦", "▪", "■", "●", "❖", "-", "–", "—", "*", "·", "\uf0b7", "\uf0a7"})
DESC_VERBS = frozenset({
    "worked", "applied", "developed", "engineered", "championed", "implemented",
    "built", "participated", "prepared", "collaborated", "automated", "optimized",
    "leveraging", "utilizing", "designed", "created", "managed", "led", "supported",
    "assisted", "maintained", "delivered", "analyzed", "conducted", "performed",
})
SUBENTRY_KEYWORDS = frozenset({"domain", "platform", "project"})


def _line_key(token: dict) -> tuple:
    return (token.get("page", 0), token.get("lineIndex", 0))


def _line_text(tokens: list[dict]) -> str:
    return " ".join(t.get("token", "") for t in tokens).strip()


def _first_meaningful_token(tokens: list[dict]) -> str:
    for t in tokens:
        tok = (t.get("token") or "").strip()
        if tok and tok not in {'"', "'", ",", "|"}:
            return tok
    return ""


def _line_has_date_anchor(tokens: list[dict]) -> bool:
    text = _line_text(tokens)
    if has_date_anchor(text):
        return True
    return any(is_date_token(t.get("token", "")) for t in tokens)


def is_bullet_line(tokens: list[dict]) -> bool:
    first = _first_meaningful_token(tokens)
    if not first:
        return False
    if first in BULLET_CHARS or first.startswith("("):
        return True
    return first.lower() in DESC_VERBS


def is_description_line(tokens: list[dict]) -> bool:
    text = _line_text(tokens)
    if is_bullet_line(tokens):
        return True
    if _line_has_date_anchor(tokens):
        return False
    words = [w.strip(".,;:").lower() for w in text.split() if w.strip()]
    return any(w in DESC_VERBS for w in words)


def is_subentry_line(tokens: list[dict]) -> bool:
    text = _line_text(tokens)
    lower = text.lower()
    return "|" in text and _line_has_date_anchor(tokens) and any(kw in lower for kw in SUBENTRY_KEYWORDS)


def is_internship_line(tokens: list[dict]) -> bool:
    text = _line_text(tokens).lower()
    return "internship" in text and _line_has_date_anchor(tokens) and not is_bullet_line(tokens)


def has_role_prefix_before_date(text: str) -> bool:
    """True when ≥2 non-date tokens appear before the first year/month anchor."""
    lower = text.lower()
    cut = first_date_anchor_pos(text)
    if cut is None or cut >= len(text):
        return False
    prefix = lower[:cut]
    words = [w.strip(".,;:|()—–-") for w in prefix.split() if w.strip(".,;:|()—–-")]
    role_words = [w for w in words if w not in MONTH_NAMES and not is_date_token(w)]
    return len(role_words) >= 2


def _recent_b_entry_line(
    line_key: tuple,
    prior_b_lines: list[tuple],
    *,
    gap: int = 5,
) -> bool:
    page, line_idx = line_key
    for p, li in prior_b_lines:
        if p == page and 0 < line_idx - li <= gap:
            return True
    return False


def is_company_continuation_line(tokens: list[dict]) -> bool:
    """Company + date sub-line directly under a title entry (Kishan FP pattern)."""
    text = _line_text(tokens)
    return "|" in text and _line_has_date_anchor(tokens)


def should_promote_entry_line(
    tokens: list[dict],
    line_preds: list[str],
    line_key: tuple,
    prior_b_lines: list[tuple],
    *,
    use_bio_hints: bool,
) -> bool:
    text = _line_text(tokens)
    bio_entry = use_bio_hints and any(t.get("tempBoundaryLabel") == "B-ENTRY" for t in tokens)
    if bio_entry and (_line_has_date_anchor(tokens) or is_internship_line(tokens) or is_subentry_line(tokens)):
        return True
    has_b = any(p == "B-ENTRY" for p in line_preds)
    has_i = any(p == "I-ENTRY" for p in line_preds)
    if bio_entry and is_bullet_line(tokens) and has_i and not has_b:
        return True
    if is_description_line(tokens):
        return False
    if not _line_has_date_anchor(tokens):
        return False
    if not (has_i and not has_b):
        return False
    if _recent_b_entry_line(line_key, prior_b_lines):
        return False
    if "|" in text:
        return False
    return has_role_prefix_before_date(text)


def apply_entry_boundary_postprocess(
    tokens: list[dict],
    word_preds: list[str],
    *,
    use_bio_hints: bool = True,
) -> list[str]:
    if not tokens or len(word_preds) != len(tokens):
        return word_preds

    result = list(word_preds)
    by_line: dict[tuple, list[int]] = defaultdict(list)
    for idx, tok in enumerate(tokens):
        by_line[_line_key(tok)].append(idx)

    prior_b_lines: list[tuple] = []
    for line_key, indices in sorted(by_line.items(), key=lambda x: (x[0][0], x[0][1])):
        indices.sort(key=lambda i: tokens[i].get("tokenIndex", i))
        line_toks = [tokens[i] for i in indices]
        line_preds = [result[i] for i in indices]
        text = _line_text(line_toks)
        bio_entry = use_bio_hints and any(
            t.get("tempBoundaryLabel") == "B-ENTRY" for t in line_toks
        )

        if any(p == "B-ENTRY" for p in line_preds):
            if not bio_entry and (
                is_description_line(line_toks)
                or not _line_has_date_anchor(line_toks)
                or is_company_continuation_line(line_toks)
            ):
                for idx in indices:
                    if result[idx] == "B-ENTRY":
                        result[idx] = "O"
            else:
                prior_b_lines.append(line_key)
            continue

        if should_promote_entry_line(
            line_toks, line_preds, line_key, prior_b_lines, use_bio_hints=use_bio_hints
        ):
            for idx in indices:
                if result[idx] in ("O", "I-ENTRY"):
                    result[idx] = "B-ENTRY"
                    prior_b_lines.append(line_key)
                    break

    return result


def demote_boundary_on_date_tokens(
    tokens: list[dict],
    word_preds: list[str],
) -> list[str]:
    """Demote mistaken B-ENTRY on date-field tokens (e.g. Present, July'22)."""
    if not tokens or len(word_preds) != len(tokens):
        return word_preds

    result = list(word_preds)
    for idx, (tok, pred) in enumerate(zip(tokens, result)):
        if pred != "B-ENTRY":
            continue
        field_bio = tok.get("_fieldBioLabel") or ""
        entity = field_bio[2:] if field_bio.startswith(("B-", "I-")) else ""
        if entity in ("SDATE", "EDATE"):
            result[idx] = "I-ENTRY"
            continue
        token = (tok.get("token") or "").strip()
        if is_date_token(token) and not is_bullet_line([tok]):
            result[idx] = "I-ENTRY"
    return result
