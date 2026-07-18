"""Pure line-level rules for project entry boundary post-process."""

from __future__ import annotations

import re

BULLETS = frozenset({
    "•", "*", "-", "▪", "◦", "■", "–", "—", "\uf0b7", "·", "✓", "✔", "\uf0a7",
    "●", "❖", "\uf0d8", "\u2022", "\u2023", "\u2043", "\u254b", "\u25b8",
})
DESC_STARTS = frozenset({
    "built", "implemented", "optimized", "developed", "engineered", "added",
    "automated", "designed", "created", "worked", "using", "through", "tracking",
    "tools", "utilized", "converted", "defined", "performed", "collaborated",
    "integrated", "developing", "applied", "leveraged", "introduced", "managing",
    "stabilizing", "stabilized", "transforming", "transformed", "focused",
    "focusing", "delivering", "delivered", "leading", "assisted", "handled",
    "served", "facilitated", "monitored", "prepared", "maintained", "supported",
    "tested", "testing", "deploying", "deployed", "configuring", "configured",
    "performing", "conducted", "collaborating",
})
NUMBERED_TITLE_RE = re.compile(r"^\d+\.\s+\S")
PROJECT_NUMBERED_RE = re.compile(r"^project\s+\d+\s*:", re.I)
GITHUB_LINK_RE = re.compile(r"github\s*link|published\s*link", re.I)
DATE_IN_LINE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\b"
    r"|\(\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.I,
)
_METADATA_PREFIXES = (
    "client:", "domain:", "team size:", "about the project:", "responsibilities:",
    "role:", "environment:", "technology:", "technologies:", "tools –", "tools -",
    "tools:", "duration:", "project duration:", "team:", "location:",
)
_METADATA_LABEL_RE = re.compile(
    r"^(client|domain|team\s*size|about\s+the\s+project|responsibilities|"
    r"role|environment|technolog(?:y|ies)|tools?|duration|location|tech\s+stack)\s*(?:used|details|stack|profile)?\s*:",
    re.I,
)


def _strip_bullet(text: str) -> str:
    s = text.strip()
    for b in sorted(BULLETS, key=len, reverse=True):
        if s.startswith(b):
            return s[len(b):].strip()
    if re.match(r"^[oO]\s+", s):
        return s[2:].strip()
    return s


def has_bullet(text: str) -> bool:
    s = text.strip()
    if re.match(r"^[oO]\s+\S", s):
        return True
    return any(s.startswith(b) for b in BULLETS)


def _first_word(text: str) -> str:
    core = _strip_bullet(text)
    words = core.split()
    return words[0].lower().rstrip(".,;:") if words else ""


def is_description_bullet(text: str) -> bool:
    if not has_bullet(text):
        return False
    return _first_word(text) in DESC_STARTS


def is_bullet_entry_head_line(text: str) -> bool:
    if not has_bullet(text):
        return False
    stripped = _strip_bullet(text).strip()
    if stripped.rstrip().endswith(":"):
        return True
    return bool(NUMBERED_TITLE_RE.match(stripped))


def is_project_metadata_line(text: str) -> bool:
    """QA-style metadata rows inside a project block (never entry heads)."""
    stripped = _strip_bullet(text).strip()
    if not stripped:
        return False
    if re.match(r"^(github\s*link|published\s*link|git\s*hub\s*link|github|link)$", stripped, re.I):
        return True
    if stripped.startswith("http") or "://" in stripped:
        return True
    lower = stripped.lower()
    if lower.startswith("responsibilities"):
        return True
    if lower.startswith("project client") or lower.startswith("projects client"):
        return True
    if re.search(r"\b(ltd|limited|inc|corp|corporation|llc|pvt)\b", lower):
        return True
    if any(lower.startswith(p) for p in _METADATA_PREFIXES):
        return True
    if _METADATA_LABEL_RE.match(stripped):
        return True
    if lower.startswith("• tools") or lower.startswith("tools –") or lower.startswith("tools -"):
        return True
    if stripped.rstrip().endswith(":") and not is_bullet_entry_head_line(text):
        head = stripped.rstrip(":").strip().lower()
        if head in {
            "client", "domain", "responsibilities", "about the project",
            "team size", "role", "environment", "technologies", "tools",
            "duration", "project duration", "location", "software type",
        }:
            return True
    # "Client : Foo" embedded in longer line
    if re.match(r"^(client|domain|team\s*size|responsibilities)\s*:", stripped, re.I):
        return True
    return False


def is_numbered_project_title(text: str) -> bool:
    return bool(PROJECT_NUMBERED_RE.match(_strip_bullet(text).strip()))


def is_link_or_date_title_line(text: str) -> bool:
    if not text.strip() or has_bullet(text) or is_description_bullet(text):
        return False
    if is_project_metadata_line(text):
        return False
    core = _strip_bullet(text).strip()
    if GITHUB_LINK_RE.search(core):
        if re.match(r"^(github\s*link|published\s*link|git\s*hub\s*link|github|link)$", core, re.I):
            return False
        if core.startswith("http") or "://" in core:
            return False
        return True
    if DATE_IN_LINE_RE.search(core) and _first_word(text) not in DESC_STARTS:
        return True
    return False


def should_suppress_boundary(text: str) -> bool:
    return is_project_metadata_line(text) or is_description_bullet(text)


def should_promote_boundary(text: str) -> bool:
    if is_bullet_entry_head_line(text):
        return True
    if is_numbered_project_title(text):
        return True
    if is_link_or_date_title_line(text):
        return True
    return False


PROJECT_NOUNS = frozenset({
    "system", "app", "application", "platform", "portal", "module", "engine",
    "tracker", "dashboard", "website", "tool", "api", "project", "automation",
    "management", "booking", "intranet", "verification", "assignment", "page",
})

METRIC_RE = re.compile(
    r"\b\d+[kK]?\+?\b.*\b(users?|requests?|daily)\b|^\d+[kK]?\s*$",
    re.I,
)


def is_project_title_line(text: str) -> bool:
    if should_promote_boundary(text):
        return True
    if is_description_bullet(text):
        return False
    stripped = _strip_bullet(text)
    if not stripped:
        return False
    if stripped.rstrip().endswith(":"):
        return has_bullet(text) or NUMBERED_TITLE_RE.match(stripped)
    if NUMBERED_TITLE_RE.match(stripped):
        return True
    lower = stripped.lower()
    if "|" in text and "github" in lower and has_bullet(text):
        return True
    title_clause = re.split(r"\s[–—]\s|\s-\s", stripped, maxsplit=1)[0]
    if has_bullet(text) and any(n in title_clause.lower() for n in PROJECT_NOUNS):
        # Require a structural separator (like -, |, or ending with :) for bulleted project titles
        has_separator = any(sep in stripped for sep in ("-", "–", "—", "|", ":"))
        if has_separator:
            return True
    return False


def is_continuation_fragment(text: str) -> bool:
    stripped = text.strip()
    if not stripped or is_description_bullet(stripped):
        return True
    if has_bullet(stripped) and is_project_title_line(stripped):
        return False
    core = _strip_bullet(stripped)
    words = core.split()
    if words and words[0][0].islower():
        return True
    if METRIC_RE.search(stripped):
        return True
    return _first_word(stripped) in DESC_STARTS

