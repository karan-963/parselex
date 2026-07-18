"""Data utilities for experience_phase3_classify: block splitting and spatial feature extraction.

Mirrors the block-building logic from:
  training_pipeline/experience/phase3_segment_classification/entry_block_dataset.py
"""

from __future__ import annotations

from inference_v2.experience_phase2_divider.date_patterns import MONTH_NAMES, is_date_token, has_date_anchor
from inference_v2.experience_phase1_segment.gap_heuristic import DATE_END_KEYWORDS

from .config import BIO_TO_CLASS, DELIMITERS


# ── Block label helpers ────────────────────────────────────────────────────────

def bio_to_class(bio: str) -> str | None:
    return BIO_TO_CLASS.get(bio or "O")


def _block_has_comp_loc(block: list[dict]) -> bool:
    return any("COMP_LOC" in (t.get("bioLabel") or "") for t in block)


def _should_split_on_delimiter(current: list[dict], token: str, bio: str) -> bool:
    if token not in DELIMITERS:
        return False
    if bio != "O":
        return False
    if _block_has_comp_loc(current):
        return False
    return True


def _bio_boundary_split(current: list[dict], bio: str) -> bool:
    """Return True if the current token's B- tag switches macro-class from current block."""
    if not current or not bio.startswith("B-"):
        return False
    new_cls = bio_to_class(bio)
    if not new_cls:
        return False
    for pt in current:
        pb = pt.get("bioLabel") or "O"
        if pb.startswith("B-"):
            prev_cls = bio_to_class(pb)
            return bool(prev_cls and prev_cls != new_cls)
    return False


def _line_boundary_split(t: dict, t_prev: dict | None) -> bool:
    if not t_prev:
        return False
    return (
        t.get("page") != t_prev.get("page")
        or t.get("lineIndex") != t_prev.get("lineIndex")
    )


def _starts_date_after_open_paren(entry_toks: list[dict], idx: int) -> bool:
    if idx <= 0:
        return False
    prev_tok = (entry_toks[idx - 1].get("token") or "").strip()
    if prev_tok != "(":
        return False
    return is_date_token(entry_toks[idx].get("token", ""))


def _should_skip_b_seg_split(entry_toks: list[dict], idx: int) -> bool:
    """Keep month-year + Present / July'22 tails in one date block."""
    tok = entry_toks[idx]
    if tok.get("segLabel") != "B-SEG":
        return False
    token = (tok.get("token") or "").strip()
    token_lower = token.lower()
    if not (is_date_token(token) or token_lower in DATE_END_KEYWORDS):
        return False
    line_key = (tok.get("page"), tok.get("lineIndex"))
    for j in range(idx - 1, -1, -1):
        prev = entry_toks[j]
        if (prev.get("page"), prev.get("lineIndex")) != line_key:
            break
        if is_date_token(prev.get("token", "")):
            return True
    return False


def split_entry_blocks(entry_toks: list[dict]) -> list[list[dict]]:
    """Split a single job entry's tokens into phrase blocks using segLabel (B-SEG/I-SEG).

    When segLabel is present, B-SEG still starts a new block but we also apply the
    same line/delimiter/spatial heuristics as training (entry_block_dataset) so a
    single I-SEG span cannot glue unrelated visual lines (e.g. date tail + ◦ project).
    Falls back to heuristic delimiter splitting if segLabel is missing or all 'O'.
    """
    has_b_seg = any(t.get("segLabel") == "B-SEG" for t in entry_toks)

    blocks: list[list[dict]] = []
    current: list[dict] = []

    for idx, t in enumerate(entry_toks):
        t_prev = entry_toks[idx - 1] if idx > 0 else None
        token = t.get("token", "")
        bio = t.get("bioLabel") or "O"
        is_new = False

        if idx == 0:
            is_new = True
        elif has_b_seg:
            if t.get("segLabel") == "B-SEG" and not _should_skip_b_seg_split(entry_toks, idx):
                is_new = True
            elif _starts_date_after_open_paren(entry_toks, idx):
                is_new = True
            elif _bio_boundary_split(current, bio):
                is_new = True
            elif _should_split_on_delimiter(current, token, bio):
                is_new = True
            elif t_prev and _should_split_on_delimiter(
                current, t_prev.get("token", ""), t_prev.get("bioLabel") or "O"
            ):
                is_new = True
            elif _line_boundary_split(t, t_prev):
                is_new = True
            elif t_prev and t.get("x0", 0.0) - t_prev.get("x1", 0.0) > 40.0:
                is_new = True
        else:
            if _starts_date_after_open_paren(entry_toks, idx):
                is_new = True
            elif _bio_boundary_split(current, bio):
                is_new = True
            elif _should_split_on_delimiter(current, token, bio):
                is_new = True
            elif t_prev and _should_split_on_delimiter(
                current, t_prev.get("token", ""), t_prev.get("bioLabel") or "O"
            ):
                is_new = True
            elif _line_boundary_split(t, t_prev):
                is_new = True
            elif t_prev and t.get("x0", 0.0) - t_prev.get("x1", 0.0) > 40.0:
                is_new = True

        if is_new and current:
            blocks.append(current)
            current = []
        current.append(t)

    if current:
        blocks.append(current)

    return blocks


def _token_is_date_part(token: str) -> bool:
    t = (token or "").strip()
    if not t:
        return False
    if is_date_token(t):
        return True
    return t.lower().rstrip(".'") in MONTH_NAMES


def _is_date_fragment_block(block: list[dict]) -> bool:
    """True when a block is part of a parenthesized date range, not a role/comp/desc phrase."""
    if not block:
        return False
    first = (block[0].get("token") or "").strip()
    if first in {"•", "◦", "▪", "∗", "*", "■", "●"}:
        return False

    text = clean_block_text(block).strip()
    if not text:
        return False
    if text in {"(", ")"}:
        return True
    if has_date_anchor(text) and not any(
        w.lower() in {"project", "client", "developer", "engineer", "intern"}
        for w in text.split()
    ):
        return True

    alnum_tokens = [
        (t.get("token") or "").strip()
        for t in block
        if any(ch.isalnum() for ch in (t.get("token") or ""))
    ]
    if not alnum_tokens:
        return True

    return all(_token_is_date_part(tok) or tok in {"(", ")", "-", "–", "—"} for tok in alnum_tokens)


def merge_adjacent_date_blocks(blocks: list[list[dict]]) -> list[list[dict]]:
    """Merge consecutive date-range fragments (e.g. '(' + 'July'22 -' + 'March' 2023 )')."""
    if not blocks:
        return blocks

    merged: list[list[dict]] = []
    i = 0
    while i < len(blocks):
        if not _is_date_fragment_block(blocks[i]):
            merged.append(blocks[i])
            i += 1
            continue
        combined = list(blocks[i])
        i += 1
        while i < len(blocks) and _is_date_fragment_block(blocks[i]):
            combined.extend(blocks[i])
            i += 1
        merged.append(combined)
    return merged


def group_experience_entries(
    tokens: list[dict],
    b_entry_lines: set[tuple],
) -> list[tuple[tuple, list[dict]]]:
    """Group experience tokens into entries using predicted B-ENTRY boundary lines.

    Returns a list of (head_key, entry_tokens) tuples in reading order.
    head_key = (page, lineIndex) of the entry's first token.
    """
    sorted_heads = sorted(b_entry_lines)
    if not sorted_heads:
        return []

    entries: list[tuple[tuple, list[dict]]] = []
    current_block: list[dict] = []
    current_head: tuple | None = None

    for t in tokens:
        if t.get("bioLabel", "O") in ("B-HEADING", "I-HEADING"):
            continue
        key = (t["page"], t.get("lineIndex", t.get("line_index", 0)))
        if key in b_entry_lines and key != current_head:
            if current_block and current_head is not None:
                entries.append((current_head, current_block))
                current_block = []
            current_head = key
        if current_head is not None:
            current_block.append(t)

    if current_block and current_head is not None:
        entries.append((current_head, current_block))

    return entries


# ── Spatial feature extraction ─────────────────────────────────────────────────

def extract_12d_spatial(token: dict, text: str) -> list[float]:
    """Extract the 12-D spatial feature vector that matches training.

    Matches entry_block_dataset.extract_12d_spatial_token() exactly.
    """
    fs = float(token.get("fontSize", token.get("font_size", 9.0)))
    bold = 1.0 if token.get("isBold", token.get("is_bold", False)) else 0.0
    page = float(token.get("page", 0))
    x0 = float(token.get("x0", 0.0))
    y0 = float(token.get("y0", 0.0))
    x1 = float(token.get("x1", x0))
    y1 = float(token.get("y1", y0))
    w = max(x1 - x0, 0.0)
    h = max(y1 - y0, 0.0)
    is_all_caps = float(text.isupper() and len(text) > 2)
    bullets = {"•", "*", "-", "▪", "◦", "■", "–", "—", "\uf0b7", "·", "✓", "✔", "\uf0a7", "●"}
    has_bullet = float(any(text.strip().startswith(b) for b in bullets))
    return [
        x0 / 612.0, y0 / 792.0, x1 / 612.0, y1 / 792.0,
        w / 612.0, h / 792.0, fs / 30.0, bold,
        is_all_caps, page / 10.0, 0.0, has_bullet,
    ]


def is_punctuation_only(text: str) -> bool:
    return not any(ch.isalnum() for ch in text)


def clean_block_text(block: list[dict]) -> str:
    words = [t.get("token", "") for t in block]
    return " ".join(words).strip()
