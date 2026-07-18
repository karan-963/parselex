"""Build education phrase blocks from step 4 B-SEG / step 5 B-EDU_START labels."""

from __future__ import annotations

from typing import Callable


def coord_key(token: dict) -> tuple[int, float, float]:
    return (
        int(token.get("page", 0)),
        round(float(token.get("x0", 0.0)), 2),
        round(float(token.get("y0", 0.0)), 2),
    )


def group_education_entries(
    tokens: list[dict],
    head_lines: set[tuple[int, int]],
) -> list[tuple[tuple[int, int] | None, list[dict]]]:
    """Group EDUCATION tokens into entries using B-EDU_START y0 row bands."""
    from inference_v2.education_phase2_divider.y0_entry_groups import group_education_tokens_by_y0

    y0_groups = group_education_tokens_by_y0(tokens, head_lines)
    entries: list[tuple[tuple[int, int] | None, list[dict]]] = []
    for block in y0_groups:
        head: tuple[int, int] | None = None
        for token in block:
            key = (int(token["page"]), int(token.get("lineIndex", token.get("line_index", 0))))
            if key in head_lines:
                head = key
                break
        entries.append((head, block))
    return entries


def split_education_blocks(entry_tokens: list[dict]) -> list[list[dict]]:
    """Split one education entry into phrase blocks using step 4 segLabel (B-SEG / I-SEG)."""
    has_b_seg = any(token.get("segLabel") == "B-SEG" for token in entry_tokens)
    if not has_b_seg:
        return [entry_tokens] if entry_tokens else []

    blocks: list[list[dict]] = []
    current: list[dict] = []

    for token in entry_tokens:
        seg = token.get("segLabel", "O")
        if seg == "B-SEG":
            if current:
                blocks.append(current)
            current = [token]
        elif seg == "I-SEG":
            current.append(token)
        elif current:
            blocks.append(current)
            current = []

    if current:
        blocks.append(current)
    return blocks


def block_to_segment(
    block_tokens: list[dict],
    clean_cid_tokens: Callable[[list[dict]], list[dict]],
    construct_sentences_by_appearance: Callable[..., list[dict]],
) -> dict | None:
    """Convert a token block into a training-style segment dict."""
    if not block_tokens:
        return None

    cleaned = clean_cid_tokens(block_tokens)
    segments = construct_sentences_by_appearance(cleaned)
    if segments:
        segment = segments[0]
        segment["tokens"] = block_tokens
        return segment

    text = " ".join(token.get("token", "") for token in block_tokens).strip()
    if not text:
        return None

    x0 = min(float(token.get("x0", 0.0)) for token in block_tokens)
    y0 = min(float(token.get("y0", 0.0)) for token in block_tokens)
    x1 = max(float(token.get("x1", x0)) for token in block_tokens)
    y1 = max(float(token.get("y1", y0)) for token in block_tokens)
    font_sizes = [float(token.get("fontSize", token.get("font_size", 9.0))) for token in block_tokens]
    font_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 9.0
    bold = 1.0 if sum(1 for token in block_tokens if token.get("isBold", token.get("is_bold", False))) > len(block_tokens) / 2 else 0.0

    return {
        "text": text,
        "tokens": block_tokens,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "spatial": [font_size, bold, 0.0, float(block_tokens[0].get("page", 1)), y0, x0, y1, x1, 3.0],
    }


def build_education_block_segments(
    education_tokens: list[dict],
    head_lines: set[tuple[int, int]],
    clean_cid_tokens: Callable[[list[dict]], list[dict]],
    construct_sentences_by_appearance: Callable[..., list[dict]],
    is_education_segment: Callable[[dict], bool],
) -> list[dict]:
    """Build phrase segments from step 4 blocks inside step 5 entry boundaries."""
    ordered = sorted(
        education_tokens,
        key=lambda token: (
            int(token.get("page", 1)),
            float(token.get("y0", 0.0)),
            float(token.get("x0", 0.0)),
        ),
    )

    if not any(token.get("segLabel") == "B-SEG" for token in ordered):
        cleaned = clean_cid_tokens(ordered)
        return [
            segment
            for segment in construct_sentences_by_appearance(cleaned)
            if is_education_segment(segment)
        ]

    segments: list[dict] = []
    for _, entry_tokens in group_education_entries(ordered, head_lines):
        for block in split_education_blocks(entry_tokens):
            segment = block_to_segment(block, clean_cid_tokens, construct_sentences_by_appearance)
            if segment and is_education_segment(segment):
                segments.append(segment)
    return segments
