"""Font-style entry-head heuristic for multi-style experience sections."""

from __future__ import annotations

from collections import Counter, defaultdict

from .entry_postprocess import (
    _line_key,
    _line_text,
    has_date_anchor,
    is_description_line,
)


def _meaningful_tokens(line_toks: list[dict]) -> list[dict]:
    return [t for t in line_toks if (t.get("token") or "").strip()]


def line_style_profile(line_toks: list[dict]) -> tuple[float, bool] | None:
    """(fontSize, isBold) of the line's most prominent (largest-font) run.

    Title lines are commonly mixed-style in resumes — a bold role name
    followed by a regular-weight company name on the same line (e.g. "Lead
    AI Engineer — Kuanta AI") — so we characterize the line by its dominant
    run rather than requiring uniform style across every token.
    """
    meaningful = _meaningful_tokens(line_toks)
    if not meaningful:
        return None
    max_size = max(round(float(t.get("fontSize", 9.0)), 1) for t in meaningful)
    dominant = [t for t in meaningful if round(float(t.get("fontSize", 9.0)), 1) == max_size]
    bold = any(bool(t.get("isBold", False)) for t in dominant)
    return (max_size, bold)


def _is_title_style_line(line_toks: list[dict]) -> bool:
    text = _line_text(line_toks)
    if not text or is_description_line(line_toks):
        return False
    if "|" in text and has_date_anchor(text):
        return False
    return True


def apply_style_entry_heuristic(tokens: list[dict], word_preds: list[str]) -> list[str]:
    if not tokens or len(word_preds) != len(tokens):
        return word_preds

    by_line: dict[tuple, list[int]] = defaultdict(list)
    for idx, tok in enumerate(tokens):
        if tok.get("section") != "EXPERIENCE":
            continue
        if tok.get("bioLabel") in ("B-HEADING", "I-HEADING"):
            continue
        by_line[_line_key(tok)].append(idx)

    if not by_line:
        return word_preds

    ordered_keys = sorted(by_line.keys(), key=lambda k: (k[0], k[1]))
    profiles: list[tuple[float, bool] | None] = []
    line_toks_list: list[list[dict]] = []
    for key in ordered_keys:
        indices = sorted(by_line[key], key=lambda i: tokens[i].get("tokenIndex", i))
        line_toks = [tokens[i] for i in indices]
        line_toks_list.append(line_toks)
        profiles.append(line_style_profile(line_toks))

    unique_profiles = {p for p in profiles if p is not None}
    if len(unique_profiles) <= 1:
        return word_preds

    # Title style = the most common bold profile (falls back to the most
    # common profile overall if nothing is bold). Using the mode instead of
    # "first line in document order" avoids anchoring on a stray body line
    # that happens to have uniform style before the first real title line.
    non_none = [p for p in profiles if p is not None]
    if not non_none:
        return word_preds
    bold_profiles = [p for p in non_none if p[1]]
    ref = Counter(bold_profiles).most_common(1)[0][0] if bold_profiles else Counter(non_none).most_common(1)[0][0]

    ref_size, ref_bold = ref
    result = list(word_preds)
    for key, line_toks, profile in zip(ordered_keys, line_toks_list, profiles):
        indices = sorted(by_line[key], key=lambda i: tokens[i].get("tokenIndex", i))
        already_b_entry = any(result[i] == "B-ENTRY" for i in indices)

        if profile != ref and profile is not None and already_b_entry:
            size, bold = profile
            # Style mismatch: smaller font and/or not bold relative to the
            # established title style — the model false-positived on a
            # body/description line, demote it back to O.
            if size < ref_size and (ref_bold and not bold):
                for idx in indices:
                    if result[idx] == "B-ENTRY":
                        result[idx] = "O"
            continue

        if profile != ref or not _is_title_style_line(line_toks):
            continue
        if already_b_entry:
            continue
        for idx in indices:
            if result[idx] in ("O", "I-ENTRY"):
                result[idx] = "B-ENTRY"
                break
    return result
