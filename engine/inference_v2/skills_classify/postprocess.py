"""Inference-only heuristics after model argmax (comma-list gaps, BIO continuity)."""

from __future__ import annotations

import re

_SKILL_LABELS = frozenset({"B-SKILL", "I-SKILL", "B-SKILL_TYPE", "I-SKILL_TYPE"})


def _line_key(token: dict) -> tuple[int, int]:
    return (int(token.get("page", 0)), int(token.get("lineIndex", 0)))


def _token_text(token: dict) -> str:
    return (token.get("token") or "").strip()


def _is_alphanumeric(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z0-9]", text))


def _is_punctuation_only(text: str) -> bool:
    return bool(text) and not _is_alphanumeric(text)


def _is_skill_label(label: str) -> bool:
    return label in _SKILL_LABELS


def _prev_alnum_index(tokens: list[dict], labels: list[str], index: int) -> int | None:
    for back in range(index - 1, -1, -1):
        text = _token_text(tokens[back])
        if not text:
            continue
        if _is_punctuation_only(text):
            continue
        return back
    return None


def _line_leading_punctuation_continuation(tokens: list[dict], labels: list[str]) -> list[str]:
    """B-SKILL after line-leading bullets → I-SKILL (e.g. • Concept)."""
    result = list(labels)
    for index, token in enumerate(tokens):
        if token.get("section") != "SKILLS" or result[index] != "B-SKILL":
            continue
        line = _line_key(token)
        line_tokens_before = [
            tokens[prior]
            for prior in range(index)
            if _line_key(tokens[prior]) == line
        ]
        if not line_tokens_before:
            continue
        if all(_is_punctuation_only(_token_text(prior)) or not _token_text(prior) for prior in line_tokens_before):
            result[index] = "I-SKILL"
    return result


def _line_ends_with_closing_paren(tokens: list[dict], line: tuple[int, int]) -> bool:
    last_alnum_idx: int | None = None
    for index, token in enumerate(tokens):
        if _line_key(token) != line:
            continue
        if _is_alphanumeric(_token_text(token)):
            last_alnum_idx = index
    if last_alnum_idx is None:
        return False
    for index in range(last_alnum_idx + 1, len(tokens)):
        if _line_key(tokens[index]) != line:
            break
        if ")" in _token_text(tokens[index]):
            return True
    return False


def _cross_line_list_continuation(tokens: list[dict], labels: list[str]) -> list[str]:
    """First skill on a new line continuing an open list → I-SKILL (e.g. AWS after JSON)."""
    result = list(labels)
    for index, token in enumerate(tokens):
        if token.get("section") != "SKILLS" or result[index] != "B-SKILL":
            continue
        if not _is_alphanumeric(_token_text(token)):
            continue

        line = _line_key(token)
        if any(
            _line_key(tokens[prior]) == line and _is_alphanumeric(_token_text(tokens[prior]))
            for prior in range(index)
        ):
            continue

        prev_idx = _prev_alnum_index(tokens, result, index)
        if prev_idx is None or _line_key(tokens[prev_idx]) == line:
            continue
        if not _is_skill_label(result[prev_idx]):
            continue
        if _line_ends_with_closing_paren(tokens, _line_key(tokens[prev_idx])):
            continue

        gap_ok = True
        for mid in range(prev_idx + 1, index):
            if not _is_punctuation_only(_token_text(tokens[mid])) and _token_text(tokens[mid]):
                gap_ok = False
                break
        if gap_ok:
            result[index] = "I-SKILL"
    return result


def _promote_comma_list_gaps(tokens: list[dict], labels: list[str]) -> list[str]:
    """Fill O predictions on comma-separated skill tails (e.g. Git, YAML, JSON)."""
    result = list(labels)
    for index, token in enumerate(tokens):
        if token.get("section") != "SKILLS":
            continue
        text = _token_text(token)
        if not _is_alphanumeric(text) or result[index] != "O":
            continue

        line = _line_key(token)
        prev_skill_idx: int | None = None
        for back in range(index - 1, -1, -1):
            if _line_key(tokens[back]) != line:
                break
            back_text = _token_text(tokens[back])
            if _is_punctuation_only(back_text):
                continue
            if _is_skill_label(result[back]):
                prev_skill_idx = back
                break
            if _is_alphanumeric(back_text) and result[back] == "O":
                break

        if prev_skill_idx is None:
            continue

        gap_ok = True
        for mid in range(prev_skill_idx + 1, index):
            if _line_key(tokens[mid]) != line:
                gap_ok = False
                break
            mid_text = _token_text(tokens[mid])
            if not mid_text:
                continue
            if _is_punctuation_only(mid_text):
                continue
            if not _is_skill_label(result[mid]):
                gap_ok = False
                break

        if not gap_ok:
            continue

        prev_text = _token_text(tokens[prev_skill_idx]).lower()
        if text.lower() == "hub" and prev_text == "git":
            result[index] = "I-SKILL"
        else:
            result[index] = "B-SKILL"

    return result


def postprocess_skill_predictions(tokens: list[dict], predictions: list[str]) -> list[str]:
    """Apply promote-only heuristics on top of model argmax labels."""
    if len(predictions) != len(tokens):
        return list(predictions)

    labels = list(predictions)
    labels = _promote_comma_list_gaps(tokens, labels)
    labels = _line_leading_punctuation_continuation(tokens, labels)
    labels = _cross_line_list_continuation(tokens, labels)
    return labels
