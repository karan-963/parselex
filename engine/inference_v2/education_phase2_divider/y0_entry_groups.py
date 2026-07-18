"""Group education tokens by physical table row (y0), not PDF lineIndex order."""

from __future__ import annotations

Y0_TOLERANCE = 3.0


def head_y0_bands(
    tokens: list[dict],
    head_lines: set[tuple[int, int]],
) -> list[tuple[int, float]]:
    """Map B-EDU_START lines to sorted (page, y0) row anchors."""
    bands: list[tuple[int, float]] = []
    for token in tokens:
        page = int(token.get("page", 0))
        line = int(token.get("lineIndex", token.get("line_index", 0)))
        if (page, line) not in head_lines and not token.get("_eduEntryHead"):
            continue
        bands.append((page, float(token.get("y0", 0.0))))
    return sorted(set(bands), key=lambda item: (item[0], item[1]))


def group_education_tokens_by_y0(
    tokens: list[dict],
    head_lines: set[tuple[int, int]],
    *,
    y0_tol: float = Y0_TOLERANCE,
) -> list[list[dict]]:
    """Slice EDUCATION tokens into entries using y0 row bands (table-safe)."""
    bands = head_y0_bands(tokens, head_lines)
    if not bands:
        return [tokens]

    entries: list[list[dict]] = []
    for index, (page, y0_start) in enumerate(bands):
        y_lo = y0_start - y0_tol
        if index + 1 < len(bands):
            next_page, next_y0 = bands[index + 1]
            y_hi = next_y0 - (y0_tol / 2) if next_page == page else float("inf")
        else:
            y_hi = float("inf")

        row_tokens = [
            token for token in tokens
            if int(token.get("page", 0)) == page
            and y_lo <= float(token.get("y0", 0.0)) < y_hi
        ]
        if row_tokens:
            entries.append(
                sorted(row_tokens, key=lambda token: (float(token.get("y0", 0)), float(token.get("x0", 0))))
            )

    return entries or [tokens]
