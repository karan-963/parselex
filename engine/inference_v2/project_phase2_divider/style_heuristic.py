"""Bold/uniform title-line and style-alignment heuristic for PROJECT entry boundaries."""

from __future__ import annotations

import re

from .line_utils import token_text

_SECTION_HEADERS = frozenset({
    "project", "projects", "project experience", "academic project",
    "personal projects", "key projects", "selected projects",
    "project highlights", "projects highlights",
})


def _is_section_header(text: str) -> bool:
    normalized = text.strip().rstrip(":.,;").lower()
    if normalized in _SECTION_HEADERS:
        return True
    if "project" in normalized and normalized.endswith("highlights"):
        return True
    if "projects client" in normalized or "project client" in normalized:
        return True
    if re.search(r"\b(mini|academic|personal|key|selected|recent|course|minor|major)\s+projects?\b", normalized):
        return True
    if normalized.startswith("projects") and len(normalized) < 40 and not has_bullet(text):
        return True
    if normalized in ("projects", "project", "projects:"):
        return True
    return False


def has_bullet(text: str) -> bool:
    from .boundary_postprocess import has_bullet as hb
    return hb(text)


def get_line_style(group: list[int], segments: list) -> tuple[float, bool, float] | None:
    """Return (fontSize, isBold, x0) for the entry title leading this line.

    Uses the leading alphanumeric token's style rather than a majority vote:
    entry titles like "Resume TLM — Advanced NLP ..." mix a large/bold title with
    a longer normal-weight description, so a majority vote misreads the line as
    body style and disables the bold-title anchor promotion.
    """
    if not group:
        return None
    first_seg = segments[group[0]]
    tokens = [t for t in first_seg.get("tokens", []) if t.get("token", "").strip()]
    if not tokens:
        return None
    x0s = [float(t.get("x0", 0.0)) for t in tokens]
    x0 = min(x0s) if x0s else 0.0

    lead = next(
        (t for t in tokens if any(c.isalnum() for c in (t.get("token") or ""))),
        tokens[0],
    )
    size = round(float(lead.get("fontSize", 9.0)), 1)
    bold = bool(lead.get("isBold", False))
    return (size, bold, x0)


def is_valid_anchor_line(text: str) -> bool:
    if _is_section_header(text):
        return False
    from .boundary_postprocess import is_description_bullet, is_continuation_fragment
    from .boundary_line_rules import is_project_metadata_line
    if is_description_bullet(text) or is_project_metadata_line(text) or is_continuation_fragment(text):
        return False
    return True


def apply_project_style_heuristic(
    segments: list,
    seg_preds: list[str],
    groups: list[list[int]],
    line_text_by_coord: dict[tuple[int, int], str],
    *,
    is_project_segment,
) -> list[str]:
    """
    Unified project style-alignment heuristic:
    1. Find the first project-start style anchor.
       If the model predicted any B-PROJ_START, we use the first predicted line as the style anchor.
       If the model predicted nothing, we fall back to the first non-header, non-description line in the PROJECT section.
    2. Extract the style signature of this anchor: (fontSize, isBold, x0).
    3. Filter/Correct predictions:
       - Promote any line matching this style signature.
       - Suppress any prediction that does not match this style signature.
    """
    # Find anchor group
    anchor_group = None
    for g in groups:
        if not any(is_project_segment(segments[idx]) for idx in g):
            continue
        first_idx = g[0]
        if seg_preds[first_idx] == "B-PROJ_START":
            # Get line text
            line_toks = []
            for idx in g:
                line_toks.extend(segments[idx].get("tokens", []))
            line_toks.sort(key=lambda t: (t.get("tokenIndex", 0), t.get("x0", 0.0)))
            text = " ".join(t.get("token", "") for t in line_toks if t.get("token", "").strip()).strip()

            if not is_valid_anchor_line(text):
                continue
            anchor_group = g
            break

    if anchor_group is None:
        for g in groups:
            if not any(is_project_segment(segments[idx]) for idx in g):
                continue
            # Get line text
            line_toks = []
            for idx in g:
                line_toks.extend(segments[idx].get("tokens", []))
            line_toks.sort(key=lambda t: (t.get("tokenIndex", 0), t.get("x0", 0.0)))
            text = " ".join(t.get("token", "") for t in line_toks if t.get("token", "").strip()).strip()

            if not is_valid_anchor_line(text):
                continue

            anchor_group = g
            break

    if anchor_group is None:
        return seg_preds

    anchor_style = get_line_style(anchor_group, segments)
    if anchor_style is None:
        return seg_preds

    size_ref, bold_ref, x0_ref = anchor_style

    out = list(seg_preds)

    # Scenario 1: Anchor is bold -> We clear predictions in this section first to strictly enforce style-matches
    if bold_ref:
        for idx in range(len(out)):
            if is_project_segment(segments[idx]):
                out[idx] = "O"

    for g in groups:
        if not any(is_project_segment(segments[idx]) for idx in g):
            continue
        first_idx = g[0]
        line_toks = []
        for idx in g:
            line_toks.extend(segments[idx].get("tokens", []))
        line_toks.sort(key=lambda t: (t.get("tokenIndex", 0), t.get("x0", 0.0)))
        text = " ".join(t.get("token", "") for t in line_toks if t.get("token", "").strip()).strip()

        if _is_section_header(text):
            out[first_idx] = "O"
            continue

        from .boundary_postprocess import is_description_bullet, is_continuation_fragment
        from .boundary_line_rules import is_project_metadata_line
        is_bad_line = (
            is_description_bullet(text)
            or is_project_metadata_line(text)
            or is_continuation_fragment(text)
        )

        if is_bad_line:
            out[first_idx] = "O"
            continue

        curr_style = get_line_style(g, segments)
        if curr_style is None:
            continue
        size_curr, bold_curr, x0_curr = curr_style

        # Match condition
        style_matches = (
            abs(size_curr - size_ref) <= 0.5
            and bold_curr == bold_ref
            and abs(x0_curr - x0_ref) <= 10.0
        )

        # Scenario 1: Anchor is bold -> Promote matching styles
        if bold_ref:
            if style_matches and not is_bad_line:
                out[first_idx] = "B-PROJ_START"
        # Scenario 2: Anchor is NOT bold -> Standard text style. Keep model predictions but do not promote.
        else:
            pass

    return out
