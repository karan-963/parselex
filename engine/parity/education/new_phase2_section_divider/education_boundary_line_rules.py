"""Pure line-level rules for education entry boundary post-process."""

from __future__ import annotations

import re

BULLETS = frozenset({
    "•", "*", "-", "▪", "◦", "■", "–", "—", "\uf0b7", "·", "✓", "✔", "\uf0a7",
    "●", "❖", "\uf0d8", "\u2022", "\u2023", "\u2043", "\u254b", "\u25b8",
})
DEGREE_KEYWORDS = frozenset({
    "bachelor", "master", "b.tech", "btech", "m.tech", "mtech", "mba", "ph.d", "phd",
    "diploma", "secondary", "higher", "intermediate", "matriculation", "ssc", "hsc",
    "b.e", "be", "b.sc", "bsc", "m.sc", "msc", "b.a", "ba", "m.a", "ma", "b.com", "bcom",
    "degree", "doctorate", "doctoral",
})
DESC_STARTS = frozenset({
    "studied", "completed", "graduated", "pursued", "achieved", "secured",
    "relevant", "coursework", "activities", "projects", "awards", "honors",
})
METADATA_PREFIXES = (
    "cgpa:", "gpa:", "grade:", "percentage:", "marks:", "score:", "class:",
    "relevant coursework:", "coursework:", "activities:", "honors:",
)
INSTITUTION_KEYWORDS = frozenset({
    "university", "college", "institute", "institution", "school", "academy",
    "polytechnic", "hss", "h.s.s", "hs", "ssc", "sslc", "board", "campus",
})
QUALIFICATION_TOKENS = frozenset({
    "bca", "b.tech", "btech", "m.tech", "mtech", "b.e", "be", "b.sc", "bsc",
    "m.sc", "msc", "b.a", "ba", "m.a", "ma", "b.com", "bcom", "mba", "mca",
    "diploma", "sslc", "ssc", "hsc", "higher", "secondary", "intermediate",
    "matriculation", "ph.d", "phd", "bachelor", "master",
})
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DATE_IN_LINE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*['\"]?\d{2,4}\b"
    r"|\b(19|20)\d{2}\b",
    re.I,
)
_CERTIFICATE_RE = re.compile(
    r"\b(hsc|ssc|sslc|higher\s+secondary|secondary\s+school|senior\s+secondary)\b",
    re.I,
)
_EXAM_LEVEL_RE = re.compile(r"^\d+(?:st|nd|rd|th)\b", re.I)
_DEGREE_ABBREV_RE = re.compile(
    r"\b("
    r"b\.?\s*e\.?[a-z]*|b\.?\s*tech|m\.?\s*tech|b\.?\s*sc|m\.?\s*sc|"
    r"b\.?\s*com|b\.?\s*a|m\.?\s*a|m\.?\s*sc|bca|mca|mba|ph\.?\s*d"
    r")\b",
    re.I,
)
_DECORATIVE_CHARS = frozenset("—–-_=•·.| ")
_SECTION_HEADER_PHRASES = frozenset({
    "education", "academic", "academics", "qualification", "qualifications",
    "educational background", "academic background", "academic qualifications",
    "academic details", "academic profile", "academic qualification",
    "educational qualifications", "educational details", "academic record",
    "academic credentials", "scholastic profile",
})
_TABLE_HEADER_KEYWORDS = frozenset({
    "examination", "university", "school", "year", "marks", "grade",
    "percentage", "board", "course", "institution", "qualification",
})


def _strip_bullet(text: str) -> str:
    s = text.strip()
    for b in sorted(BULLETS, key=len, reverse=True):
        if s.startswith(b):
            return s[len(b):].strip()
    return s


def has_bullet(text: str) -> bool:
    s = text.strip()
    return any(s.startswith(b) for b in BULLETS)


def _first_word(text: str) -> str:
    core = _strip_bullet(text)
    words = core.split()
    return words[0].lower().rstrip(".,;:") if words else ""


def is_description_bullet(text: str) -> bool:
    if not has_bullet(text):
        return False
    return _first_word(text) in DESC_STARTS


def is_decorative_separator_line(text: str) -> bool:
    """Horizontal rules / dash dividers with no semantic content."""
    clean = text.strip()
    if not clean:
        return True
    if len(clean) >= 6 and all(ch in _DECORATIVE_CHARS for ch in clean):
        return True
    return False


def is_table_header_line(text: str) -> bool:
    """Tabular education layouts: column header row without entry data."""
    if is_decorative_separator_line(text):
        return False
    lower = _strip_bullet(text).lower()
    if not lower:
        return False
    keyword_hits = sum(1 for k in _TABLE_HEADER_KEYWORDS if k in lower)
    if keyword_hits >= 3 and not YEAR_RE.search(text) and "%" not in text:
        return True
    if "examination" in lower and ("marks" in lower or "year" in lower or "school" in lower):
        return True
    return False


def is_exam_level_line(text: str) -> bool:
    return bool(_EXAM_LEVEL_RE.match(_strip_bullet(text).strip()))


def is_certificate_qualification_line(text: str) -> bool:
    lower = _strip_bullet(text).lower()
    if _CERTIFICATE_RE.search(lower):
        return True
    if "certificate" in lower and any(k in lower for k in ("secondary", "school", "hsc", "ssc")):
        return True
    return False


def is_qualification_title_only_line(text: str) -> bool:
    """Degree/certificate row without institution name on the same line."""
    if (
        is_section_header_line(text)
        or is_pure_metadata_line(text)
        or is_decorative_separator_line(text)
    ):
        return False
    lower = _strip_bullet(text).lower()
    if any(k in lower for k in INSTITUTION_KEYWORDS):
        return False
    return (
        is_certificate_qualification_line(text)
        or is_degree_opener_line(text)
        or (is_degree_qualification_line(text) and not has_date_anchor(text))
    )


def has_degree_abbreviation(text: str) -> bool:
    return bool(_DEGREE_ABBREV_RE.search(_strip_bullet(text)))


def is_continuation_fragment_line(text: str) -> bool:
    """Wrapped table/degree tail (e.g. closing paren on its own line)."""
    stripped = _strip_bullet(text).strip()
    if not stripped or len(stripped.split()) > 4:
        return False
    if stripped.startswith((")", "]", "}", ",")):
        return True
    return stripped.endswith(")") and not any(k in stripped.lower() for k in INSTITUTION_KEYWORDS)


def is_pure_metadata_line(text: str) -> bool:
    """Standalone CGPA/GPA/grade line with no institution name."""
    lower = text.strip().lower()
    if any(lower.startswith(p) for p in METADATA_PREFIXES):
        return True
    if lower.startswith(("cgpa", "gpa", "grade", "percentage", "marks", "score")):
        if not any(k in lower for k in INSTITUTION_KEYWORDS):
            return True
    return False


def is_institution_entry_line(text: str) -> bool:
    """Institution row used as GT entry head (school/university + optional percentage)."""
    lower = _strip_bullet(text).lower()
    if (
        is_section_header_line(text)
        or is_description_bullet(text)
        or is_decorative_separator_line(text)
        or is_table_header_line(text)
        or is_continuation_fragment_line(text)
    ):
        return False
    if not any(k in lower for k in INSTITUTION_KEYWORDS):
        return False
    if is_pure_metadata_line(text):
        return False
    return (
        is_exam_level_line(text)
        or has_degree_abbreviation(text)
        or has_date_anchor(text)
        or "%" in text
    )


def is_degree_qualification_line(text: str) -> bool:
    """Degree/qualification row, often with a date range on the same line."""
    if (
        is_section_header_line(text)
        or is_pure_metadata_line(text)
        or is_decorative_separator_line(text)
        or is_table_header_line(text)
        or is_continuation_fragment_line(text)
    ):
        return False
    first = _first_word(text)
    if first in QUALIFICATION_TOKENS or first in DEGREE_KEYWORDS:
        return True
    if is_exam_level_line(text):
        return True
    if has_degree_abbreviation(text):
        return True
    if has_date_anchor(text) and not is_table_header_line(text):
        return True
    return is_degree_opener_line(text)


def is_education_metadata_line(text: str) -> bool:
    return is_pure_metadata_line(text)


def is_degree_opener_line(text: str) -> bool:
    core = _strip_bullet(text).lower()
    first = _first_word(text)
    if first in DEGREE_KEYWORDS:
        return True
    if has_degree_abbreviation(text):
        return True
    return any(core.startswith(kw) for kw in ("bachelor of", "master of", "doctor of"))


def has_date_anchor(text: str) -> bool:
    return bool(DATE_IN_LINE_RE.search(text) or YEAR_RE.search(text))


def is_section_header_line(text: str) -> bool:
    normalized = text.strip().lower().rstrip(":")
    return normalized in _SECTION_HEADER_PHRASES


def is_split_institution_line(text: str) -> bool:
    """Institution-only row where degree/qualification is expected on the next line."""
    if not is_institution_entry_line(text):
        return False
    if is_exam_level_line(text) or has_degree_abbreviation(text) or is_degree_opener_line(text):
        return False
    return True


def is_entry_head_candidate(text: str) -> bool:
    """Physical line that may carry an education entry boundary."""
    if (
        is_section_header_line(text)
        or is_decorative_separator_line(text)
        or is_table_header_line(text)
        or is_continuation_fragment_line(text)
        or is_pure_metadata_line(text)
        or is_description_bullet(text)
    ):
        return False
    return (
        is_institution_entry_line(text)
        or is_degree_qualification_line(text)
        or is_degree_opener_line(text)
        or is_certificate_qualification_line(text)
        or is_qualification_title_only_line(text)
    )


def should_suppress_boundary(text: str) -> bool:
    if is_section_header_line(text):
        return True
    if is_decorative_separator_line(text):
        return True
    if is_table_header_line(text):
        return True
    if is_continuation_fragment_line(text):
        return True
    if is_pure_metadata_line(text):
        return True
    if is_description_bullet(text):
        return True
    if is_institution_entry_line(text):
        return False

    stripped = _strip_bullet(text).strip()
    if not stripped:
        return True
    if stripped.lower().startswith(("relevant coursework", "coursework", "activities")):
        return True
    return False


def should_promote_boundary(text: str) -> bool:
    if should_suppress_boundary(text):
        return False
    if is_institution_entry_line(text):
        return True
    if is_degree_opener_line(text):
        return True
    if is_certificate_qualification_line(text) or is_qualification_title_only_line(text):
        return True
    if is_degree_qualification_line(text):
        return False
    if has_date_anchor(text) and any(k in text.lower() for k in ("university", "college", "institute", "school")):
        return True
    if has_bullet(text) and is_degree_opener_line(text):
        return True
    return False
