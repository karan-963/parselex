"""Typography-based heading promotion (style propagation pass)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .heading_heuristics import _alpha_len, is_false_heading, plain_line, resolve_section_type
from .line_builder import LineRecord

if TYPE_CHECKING:
    pass

FONT_TOL_PT = 0.6
X0_TOL_PT = 8.0
MIN_ANCHOR_STYLE_COUNT = 2

BULLET_PREFIX_RE = re.compile(r"^[\uf0d8\u2022\u25cf\u25aa\-\*•→]+\s*", re.UNICODE)
STYLE_STOPWORDS = frozenset({"responsibilities", "responsibility", "description", "achievements"})


def _build_heading_style_profiles(
    lines: list[LineRecord],
    anchor_keys: set[tuple[int, int]],
) -> list[dict]:
    buckets: dict[str, dict] = {}
    for line in lines:
        if line.key not in anchor_keys:
            continue
        bucket_key = f"{line.font_median:.1f}_{line.is_bold}_{round(line.x0)}"
        if bucket_key not in buckets:
            buckets[bucket_key] = {
                "count": 0,
                "x0_sum": 0.0,
                "font_median": line.font_median,
                "is_bold": line.is_bold,
            }
        buckets[bucket_key]["count"] += 1
        buckets[bucket_key]["x0_sum"] += line.x0

    profiles: list[dict] = []
    for bucket in buckets.values():
        if bucket["count"] >= MIN_ANCHOR_STYLE_COUNT:
            profiles.append({
                "font_median": bucket["font_median"],
                "is_bold": bucket["is_bold"],
                "x0": bucket["x0_sum"] / bucket["count"],
            })
    return profiles


def _line_matches_heading_style(line: LineRecord, profile: dict) -> bool:
    return (
        line.is_bold == profile["is_bold"]
        and abs(line.font_median - profile["font_median"]) <= FONT_TOL_PT
        and abs(line.x0 - profile["x0"]) <= X0_TOL_PT
    )


def _is_style_propagation_candidate(
    line: LineRecord,
    plain: str,
    raw: str,
    style_match: bool,
) -> bool:
    if not plain or BULLET_PREFIX_RE.match(raw.strip()):
        return False
    if _alpha_len(raw) > 45 or len(plain.split()) > 5:
        return False
    if is_false_heading(plain, raw):
        return False
    if not style_match and re.match(r"^hobbies\s*:", plain, re.I):
        return False
    if re.match(r"^(client|role|project name|description)\s*:", plain, re.I):
        return False
    if re.match(r"^[•\-]", raw.strip()):
        return False
    if "|" in plain and len(plain) > 28:
        return False
    if plain.endswith(".") and len(plain) > 20 and resolve_section_type(plain) == "OTHER":
        return False
    if any(w in STYLE_STOPWORDS for w in plain.lower().split()):
        return False
    if line.page == 1 and line.lineIndex <= 1 and resolve_section_type(plain) == "OTHER":
        return False
    return True


def propagate_style_headings(
    lines: list[LineRecord],
    anchor_keys: set[tuple[int, int]],
    candidate_keys: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    profiles = _build_heading_style_profiles(lines, anchor_keys)
    if not profiles:
        return set()

    promoted: set[tuple[int, int]] = set()
    for line in lines:
        if line.key not in candidate_keys:
            continue
        plain = plain_line(line.text)
        raw = line.text
        for profile in profiles:
            style_match = _line_matches_heading_style(line, profile)
            if style_match and _is_style_propagation_candidate(line, plain, raw, style_match):
                promoted.add(line.key)
                break
    return promoted
