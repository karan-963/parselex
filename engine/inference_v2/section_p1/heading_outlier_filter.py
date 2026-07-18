"""Demote style-outlier headings (sub-labels mis-detected as section headings).

A bulleted, indented sub-label such as ``• Soft Skills :`` sitting inside the
SKILLS section is often promoted to a heading by the keyword heuristics. It then
splits the section twice in phase 2. This pass compares each detected heading to
the dominant heading typography and drops the ones that deviate sharply.
"""

from __future__ import annotations

import re

from .line_builder import LineRecord

BULLET_PREFIX_RE = re.compile(r"^[\uf0d8\u2022\u25cf\u25aa\-\*•→]+\s*", re.UNICODE)

X0_TOL_PT = 5.0
MIN_HEADINGS_FOR_PROFILE = 4
MIN_DOMINANT_MATCH = 3


def _plain(text: str) -> str:
    return BULLET_PREFIX_RE.sub("", text.strip()).strip()


def _has_bullet(text: str) -> bool:
    return bool(BULLET_PREFIX_RE.match(text.strip()))


def _is_caps(text: str) -> bool:
    return any(c.isalpha() for c in text) and text == text.upper()


def find_outlier_heading_keys(heading_lines: list[LineRecord]) -> set[tuple[int, int]]:
    """Return heading keys whose typography deviates sharply from the dominant style.

    Conservative by design: only fires when a strong dominant style group exists
    (multiple clean, non-bulleted headings sharing an indentation), and only
    removes a candidate that is bullet-prefixed (rare for real section headings)
    AND additionally deviates on indentation or casing.
    """
    n = len(heading_lines)
    if n < MIN_HEADINGS_FOR_PROFILE:
        return set()

    x0s = sorted(line.x0 for line in heading_lines)
    dominant_x0 = x0s[len(x0s) // 2]
    caps_majority = sum(1 for line in heading_lines if _is_caps(_plain(line.text))) > n / 2
    bullet_rare = sum(1 for line in heading_lines if _has_bullet(line.text)) <= n / 2

    # Require a clear anchor group of clean headings at the dominant indentation.
    dominant_matches = sum(
        1 for line in heading_lines
        if abs(line.x0 - dominant_x0) <= X0_TOL_PT and not _has_bullet(line.text)
    )
    if dominant_matches < MIN_DOMINANT_MATCH:
        return set()

    outliers: set[tuple[int, int]] = set()
    for line in heading_lines:
        if not (bullet_rare and _has_bullet(line.text)):
            continue
        plain = _plain(line.text)
        indent_dev = line.x0 > dominant_x0 + X0_TOL_PT
        case_dev = caps_majority and not _is_caps(plain)
        if indent_dev or case_dev:
            outliers.add(line.key)
    return outliers
