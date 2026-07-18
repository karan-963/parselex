"""Heuristic label refinement for project segment classification.

Targets the dominant failure modes (DESC ↔ PROJECT_NAME) identified in
common_failures.md without cross-section coupling.
"""

from __future__ import annotations

import re

_BULLET_PREFIX = re.compile(r"^[•\-\*●·▪■◦✓✔\uf0b7\uf0a7\s]+")

_GENERIC_HEADER = re.compile(
    r"^(?:objective|tools|status|technologies|tech\s*stack|features|"
    r"responsibilities|role|duration|description|overview|platform|"
    r"environment|summary|achievements|highlights|methodology|approach|"
    r"key\s*skills|modules|components)\s*:?\s*$",
    re.IGNORECASE,
)

_METADATA_COLON_LINE = re.compile(
    r"^(?:project\s+overview|project\s+description|technologies\s+used|"
    r"languages/?frameworks|programming\s*(?:&|and)?\s*frameworks|"
    r"languages|domain|description|algorithms?|skills?|tools?|"
    r"cloud(?:\s*(?:&|/)\s*\w+)?|data(?:\s*(?:&|/)\s*\w+)?|"
    r"orchestration(?:\s*&\s*\w+)?|fintech(?:\s+integration)?|"
    r"databases?|security|governance|compliance|api(?:\s*&\s*integration)?)\s*:",
    re.IGNORECASE,
)

_ROLE_TITLE_PIPE = re.compile(
    r"^(?:lead|senior|junior|principal|staff)?\s*"
    r"(?:[\w\s\-/&]+?\s+)?"
    r"(?:developer|engineer|programmer|consultant|architect|analyst|manager)\s*\|",
    re.IGNORECASE,
)

_NUMBERED_SUBENTRY = re.compile(r"^\d+\.\s+\S")

_STATUS_PHRASE = re.compile(
    r"^(?:in\s+progress|ongoing|completed|currently\s+in\s+development|"
    r"status\s*:\s*(?:currently|in\s+development|ongoing))",
    re.IGNORECASE,
)

_COMPANY_SUFFIX = re.compile(
    r"\b(?:limited|ltd\.?|inc\.?|corp\.?|pvt\.?|llc|technologies|solutions)\s*\.?\s*$",
    re.IGNORECASE,
)

_LINK_REFERENCE = re.compile(
    r"^(?:code\s*link|github|gitlab|demo\s*link|live\s*link|repository|"
    r"source\s*code|project\s*link)\b",
    re.IGNORECASE,
)

_PAGE_FOOTER = re.compile(r"^page\s+\d+\s+of\s+\d+$", re.IGNORECASE)

_NARRATIVE_START = re.compile(
    r"^(?:in\s+this|i\s+(?:designed|developed|built|created|implemented|used)|"
    r"used\s+to\s+show|diodes\s*,|managed\s+the|directed\s+the|architected\s+a)",
    re.IGNORECASE,
)

_TECH_PIPE_LIST = re.compile(
    r"^[\w\s\.\+\#]+(?:\s*,\s*\|\s*|\s*\|\s*)[\w\s\.\+\#]+$",
    re.IGNORECASE,
)

_LEADING_PIPE = re.compile(r"^\|")

_NUMBERED_INDEX_ONLY = re.compile(r"^\d+\.\s*$")

_DECORATIVE_CHARS = frozenset("—–-_=•·.| ")


def _strip_bullets(text: str) -> str:
    return _BULLET_PREFIX.sub("", text).strip()


def is_decorative_segment(text: str) -> bool:
    """Repeated dashes/rules or other non-semantic divider lines."""
    clean = text.strip()
    if not clean:
        return True
    if len(clean) >= 6 and all(ch in _DECORATIVE_CHARS for ch in clean):
        return True
    return bool(_NUMBERED_INDEX_ONLY.match(clean))


def should_downgrade_project_name(text: str) -> bool:
    """True when segment text is structural metadata, not a project title."""
    clean = _strip_bullets(text.strip())
    if not clean:
        return False
    if is_decorative_segment(clean):
        return True
    if _GENERIC_HEADER.match(clean):
        return True
    if _METADATA_COLON_LINE.search(clean):
        return True
    if _ROLE_TITLE_PIPE.search(clean):
        return True
    if _NUMBERED_SUBENTRY.match(clean) and len(clean.split()) <= 10:
        return True
    if _STATUS_PHRASE.search(clean):
        return True
    if _COMPANY_SUFFIX.search(clean) and len(clean.split()) <= 8:
        return True
    if _LINK_REFERENCE.match(clean):
        return True
    if _PAGE_FOOTER.match(clean):
        return True
    if _NARRATIVE_START.search(clean):
        return True
    if clean.endswith(".") and len(clean.split()) <= 6:
        return True
    if _TECH_PIPE_LIST.match(clean) and "," in clean:
        return True
    if _LEADING_PIPE.match(clean):
        return True
    if clean.count("|") >= 2:
        return True
    if len(clean.split()) <= 4 and clean[0].islower():
        return True
    if " & " in clean and len(clean.split()) <= 4 and not clean.isupper():
        return True
    return False


def refine_segment_label(text: str, label: str) -> str:
    """Apply deterministic rules to correct ambiguous PROJECT_NAME assignments."""
    if label == "PROJECT_NAME" and should_downgrade_project_name(text):
        return "DESC"
    return label
