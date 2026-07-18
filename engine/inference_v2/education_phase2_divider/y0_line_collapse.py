"""Collapse duplicate boundary lines that share the same physical y0 row."""

from __future__ import annotations

Y0_TOLERANCE = 3.0


def _line_y0(tokens: list[dict], page: int, line_index: int) -> float | None:
    line_toks = [
        t for t in tokens
        if int(t.get("page", 0)) == page
        and int(t.get("lineIndex", t.get("line_index", 0))) == line_index
    ]
    if not line_toks:
        return None
    return min(float(t.get("y0", 0.0)) for t in line_toks)


def _line_min_x0(tokens: list[dict], page: int, line_index: int) -> float:
    line_toks = [
        t for t in tokens
        if int(t.get("page", 0)) == page
        and int(t.get("lineIndex", t.get("line_index", 0))) == line_index
    ]
    if not line_toks:
        return 9999.0
    return min(float(t.get("x0", 0.0)) for t in line_toks)


def collapse_lines_by_y0(
    tokens: list[dict],
    lines: set[tuple[int, int]],
    *,
    y0_tol: float = Y0_TOLERANCE,
) -> set[tuple[int, int]]:
    """Keep one (page, lineIndex) per physical row band; prefer leftmost column."""
    if not lines:
        return set()

    bands: dict[tuple[int, float], list[tuple[int, int]]] = {}
    for page, line_idx in lines:
        y0 = _line_y0(tokens, page, line_idx)
        if y0 is None:
            continue
        band_key = (page, round(y0 / y0_tol) * y0_tol)
        bands.setdefault(band_key, []).append((page, line_idx))

    collapsed: set[tuple[int, int]] = set()
    for coords in bands.values():
        if len(coords) == 1:
            collapsed.add(coords[0])
            continue
        collapsed.add(min(coords, key=lambda c: _line_min_x0(tokens, c[0], c[1])))
    return collapsed
