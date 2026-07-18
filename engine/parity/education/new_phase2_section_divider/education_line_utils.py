"""Physical-line helpers for education phase-2 boundary eval."""

from __future__ import annotations


def token_text(tok: dict) -> str:
    return (tok.get("token") or tok.get("text") or "").strip()


def physical_line_text_from_tokens(tokens: list, page: int, line_idx: int) -> str:
    """Full line text from raw layout tokens (preserves bullets, punctuation)."""
    line_toks = [
        t for t in tokens
        if t and t.get("page") == page and t.get("lineIndex") == line_idx
    ]
    if not line_toks:
        return ""
    line_toks.sort(key=lambda t: (t.get("tokenIndex", 0), t.get("x0", 0.0)))
    return " ".join(token_text(t) for t in line_toks if token_text(t))


def physical_line_text(segments: list, page: int, line_idx: int) -> str:
    """Reconstruct full line text from segment-attached tokens."""
    toks: list[dict] = []
    for seg in segments:
        for t in seg.get("tokens", []):
            if not t:
                continue
            if t.get("page") == page and t.get("lineIndex") == line_idx:
                toks.append(t)
    if not toks:
        return ""
    toks.sort(key=lambda t: (t.get("tokenIndex", 0), t.get("x0", 0.0)))
    return " ".join(token_text(t) for t in toks if token_text(t))


def build_physical_line_text_map(
    segments: list,
    raw_tokens: list | None = None,
) -> dict[tuple[int, int], str]:
    if raw_tokens:
        coords: set[tuple[int, int]] = set()
        for t in raw_tokens:
            if t and "page" in t and "lineIndex" in t:
                coords.add((t["page"], t["lineIndex"]))
        return {
            (p, l): physical_line_text_from_tokens(raw_tokens, p, l)
            for p, l in coords
        }
    coords: set[tuple[int, int]] = set()
    for seg in segments:
        for t in seg.get("tokens", []):
            if t and "page" in t and "lineIndex" in t:
                coords.add((t["page"], t["lineIndex"]))
    return {(p, l): physical_line_text(segments, p, l) for p, l in coords}


def line_keys_for_group(group: list[int], segments: list) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for idx in group:
        for t in segments[idx].get("tokens", []):
            if t and "page" in t and "lineIndex" in t:
                keys.add((t["page"], t["lineIndex"]))
    return keys


def group_has_boundary_start(seg_preds: list[str], group: list[int]) -> bool:
    return any(seg_preds[idx] == "B-EDU_START" for idx in group)


def collapse_boundary_to_line_anchor(
    seg_preds: list[str],
    groups: list[list[int]],
    segments: list,
    is_education_segment,
) -> list[str]:
    """One B-EDU_START per line group; anchor = first in-section segment (not punct-only prefix)."""
    out = list(seg_preds)
    for group in groups:
        if not group_has_boundary_start(out, group):
            continue
        for idx in group:
            out[idx] = "O"
        anchor = next(
            (i for i in group if is_education_segment(segments[i])),
            group[0],
        )
        out[anchor] = "B-EDU_START"
    return out


def pred_line_set_from_groups(
    seg_preds: list[str],
    groups: list[list[int]],
    segments: list,
    education_lines: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Line-level: any B-EDU_START on the group marks the whole physical line."""
    lines: set[tuple[int, int]] = set()
    for group in groups:
        if not group_has_boundary_start(seg_preds, group):
            continue
        for key in line_keys_for_group(group, segments):
            if key in education_lines:
                lines.add(key)
    return lines


def assign_line_level_token_predictions(
    seg_preds: list[str],
    groups: list[list[int]],
    segments: list,
    num_segs: int,
) -> list[dict]:
    """First token of every segment on a predicted line gets B-EDU_START (whole-line mark)."""
    line_start_groups: set[int] = set()
    for gi, group in enumerate(groups):
        if group_has_boundary_start(seg_preds, group):
            line_start_groups.add(gi)

    tokens: list[dict] = []
    group_by_seg = {}
    for gi, group in enumerate(groups):
        for idx in group:
            group_by_seg[idx] = gi

    for idx in range(num_segs):
        gi = group_by_seg.get(idx)
        line_pred = gi is not None and gi in line_start_groups
        label = "B-EDU_START" if line_pred else seg_preds[idx]
        for token_i, tok in enumerate(segments[idx].get("tokens", [])):
            if tok and isinstance(tok, dict):
                tok = dict(tok)
                tok["prediction"] = label if token_i == 0 else "O"
                tokens.append(tok)
    return tokens
