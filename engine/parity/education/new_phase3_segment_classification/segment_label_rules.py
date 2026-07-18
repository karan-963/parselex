"""Heuristic label refinement for education segment classification."""

from __future__ import annotations

import re

_BULLET_PREFIX = re.compile(r"^[•\-\*●·▪■◦✓✔\uf0b7\uf0a7\s]+")

_EDUCATION_SECTION_HEADER_PHRASES = frozenset({
    "education", "academic", "academics", "qualification", "qualifications",
    "educational background", "academic background", "academic qualifications",
    "academic details", "academic profile", "academic qualification",
    "educational qualifications", "educational qualification",
    "educational details", "academic record", "academic credentials",
    "scholastic profile", "education and certifications",
    "education certifications",
})


def _normalize_section_header(text: str) -> str:
    s = re.sub(r"\s+", " ", text.strip().lower()).rstrip(":")
    return s.replace(" & ", " and ")


def is_education_section_header_text(text: str) -> bool:
    """True when segment text is the EDUCATION section title line, not entry content."""
    return _normalize_section_header(text) in _EDUCATION_SECTION_HEADER_PHRASES


def is_education_section_heading(seg: dict) -> bool:
    """Exclude section-heading segments from field classification (inference + training)."""
    seg_tokens = seg.get("tokens", [])
    if any(t.get("bioLabel") in ("B-HEADING", "I-HEADING") for t in seg_tokens):
        return True
    return is_education_section_header_text(seg.get("text", ""))

_GPA_LINE = re.compile(
    r"\b(?:cgpa|gpa|percentage|percent|marks?|score|grade\s*point)\b",
    re.IGNORECASE,
)
_LOC_LINE = re.compile(
    r"^(?:city|state|country|location|address)\s*:",
    re.IGNORECASE,
)
_METADATA_COLON = re.compile(
    r"^(?:course|programme|program|specialization|major|minor|board|stream)\s*:",
    re.IGNORECASE,
)
_DECORATIVE_CHARS = frozenset("—–-_=•·.| ")


def _strip_bullets(text: str) -> str:
    return _BULLET_PREFIX.sub("", text.strip()).strip()


def is_decorative_segment(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return True
    if len(clean) >= 6 and all(ch in _DECORATIVE_CHARS for ch in clean):
        return True
    return False


def should_downgrade_institution(text: str) -> bool:
    """True when segment text is metadata, not an institution name."""
    clean = _strip_bullets(text.strip())
    if not clean:
        return False
    if is_decorative_segment(clean):
        return True
    if _GPA_LINE.search(clean):
        return True
    if _LOC_LINE.search(clean):
        return True
    if _METADATA_COLON.search(clean):
        return True
    if clean.count("|") >= 2 and len(clean.split()) <= 8:
        return True
    return False


def should_downgrade_degree(text: str) -> bool:
    clean = _strip_bullets(text.strip())
    if not clean:
        return False
    if _GPA_LINE.search(clean) and not re.search(r"\b(?:b\.?tech|mba|m\.?sc)\b", clean, re.IGNORECASE):
        return True
    return False


def refine_segment_label(text: str, label: str) -> str:
    if label == "INSTITUTION" and should_downgrade_institution(text):
        return "DESCRIPTION"
    if label == "DEGREE" and should_downgrade_degree(text):
        return "DESCRIPTION"
    return label
