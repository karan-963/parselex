"""Line-only heading heuristics — keyword map + font/style rules (ported from v2)."""

from __future__ import annotations

import re
from typing import Any

from .line_builder import LineRecord, build_parser_lines
from .heading_postprocess import dedupe_adjacent_heading_keys

MAX_ALPHA_LEN = 45

HEADING_TO_SECTION: dict[str, str] = {
    "contact": "PERSONAL",
    "personal": "PERSONAL",
    "summary": "SUMMARY",
    "objective": "SUMMARY",
    "profile summary": "SUMMARY",
    "experience": "EXPERIENCE",
    "work experience": "EXPERIENCE",
    "professional experience": "EXPERIENCE",
    "employment": "EXPERIENCE",
    "education": "EDUCATION",
    "academia": "EDUCATION",
    "academic": "EDUCATION",
    "qualification": "EDUCATION",
    "skills": "SKILLS",
    "technical skills": "SKILLS",
    "core competencies": "SKILLS",
    "areas of expertise": "SKILLS",
    "projects": "PROJECTS",
    "certification": "CERTIFICATIONS",
    "internship": "EXPERIENCE",
    "languages": "SKILLS",
    "soft skills": "SKILLS",
    "key achievements": "OTHER",
}

EXTRA_HEADING_MAP: dict[str, str] = {
    "personal details": "PERSONAL",
    "academic qualification": "EDUCATION",
    "academic details": "EDUCATION",
    "technical qualification": "SKILLS",
    "about": "SUMMARY",
    "about me": "SUMMARY",
    "profile": "SUMMARY",
    "interests": "OTHER",
    "interest": "OTHER",
    "hobbies": "OTHER",
    "hobby": "OTHER",
    "declaration": "OTHER",
    "awards": "OTHER",
    "award": "OTHER",
    "certifications": "CERTIFICATIONS",
    "project": "PROJECTS",
    "tools & platforms": "SKILLS",
    "tools and platforms": "SKILLS",
    "tools & technologies": "SKILLS",
    "tools and technologies": "SKILLS",
    "professional summary": "SUMMARY",
    "websites": "OTHER",
    "portfolios": "OTHER",
    "work history": "EXPERIENCE",
    "organizational experience": "EXPERIENCE",
    "organisational experience": "EXPERIENCE",
    "ui/ux designing": "PROJECTS",
    "ui/ux design": "PROJECTS",
    "achievements & certification": "CERTIFICATIONS",
    "achievements and certification": "CERTIFICATIONS",
    "profiles": "OTHER",
}

ENDSWITH_BLOCKLIST = frozenset({
    "project", "profile", "summary", "skills", "education", "experience",
    "award", "interest", "about", "employment", "certification",
})

HEADING_KEYS: list[str] = sorted(
    {**HEADING_TO_SECTION, **EXTRA_HEADING_MAP}.keys(),
    key=len,
    reverse=True,
)

BULLET_PREFIX_RE = re.compile(r"^[\uf0d8\u2022\u25cf\u25aa\-\*•→]+\s*", re.UNICODE)
CONTACT_FIELD_RE = re.compile(
    r"^(name|email|phone|mobile|linkedin|github|e-mail|date of birth|gender|"
    r"institute name|qualification|board/university|percentage|e-mail id)\s*:?"
    r"|^hobbies\s*:",  # form-row label only (colon); standalone HOBBIES is a section heading
    re.I,
)
FALSE_HEADING_RE = re.compile(
    r"^(?:project\s+name|client\s*(?:description)?\s*:|role\s+description|tools\s*:|"
    r"description\s*:|applications\s*:|responsibilities\s*:|designation\s*:|"
    r"design\s+link\s*:|portfolio\s*:|reporting(?:\s+tool)?\s*:|"
    r"roles?\s+and\s+responsibilities|\d+\.?\s*project\s*name|project\d+\s*:|"
    r"experience\s+.+\(.+\))",
    re.I,
)
JOB_TITLE_RE = re.compile(
    r"\b(engineer|developer|analyst|consultant|manager|intern|architect|tester|"
    r"lead|fresher|associate|professional|scientist)\b",
    re.I,
)
DATE_ONLY_RE = re.compile(
    r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}"
    r"|\d{4}\s*[-–—]\s*(?:present|\d{4})",
    re.I,
)
EXP_ENTRY_RE = re.compile(
    r"\b(?:engineer|developer|analyst|consultant|manager|architect|designer|scientist)\b"
    r".*\d{2}\s*/\s*\d{4}\s+to\b",
    re.I,
)
DEGREE_LINE_RE = re.compile(
    r"^(?:master|bachelor|mba|m\.?\s*tech|b\.?\s*tech|m\.?\s*e\.?|b\.?\s*e\.?|ph\.?\s*d\.?)\b",
    re.I,
)
COMPACT_SECTION_RE = re.compile(
    r"^(?:organizational|organisational|professional|work|employment)?experience$",
    re.I,
)
SKILL_PIPE_RE = re.compile(r"^[A-Za-z0-9+#.\-/ ]+\|\s*$")

TYPO_FIXES = {"educaton": "education", "declaraton": "declaration", "declartaion": "declaration"}
OTHER_SECTION_HEADINGS = frozenset(
    {"awards", "award", "interests", "interest", "hobbies", "hobby",
     "declaration", "achievements", "achievement"}
)
SINGLE_WORD_HEADINGS = frozenset(
    {
        "contact", "objective", "internship", "education", "skills", "summary",
        "projects", "experience", "languages", "certification", "employment",
        "hobbies", "hobby",
    }
)
FULL_MAP: dict[str, str] = {**HEADING_TO_SECTION, **EXTRA_HEADING_MAP}


def _fuzzy_lower(text: str) -> str:
    t = text.lower()
    for typo, fix in TYPO_FIXES.items():
        t = t.replace(typo, fix)
    return t


def collapse_spaced_caps(plain: str) -> str:
    parts = plain.split()
    if len(parts) >= 4:
        single = sum(1 for p in parts if len(p) == 1 and p.isalpha())
        if single >= len(parts) * 0.75:
            return "".join(parts)
    return plain


def _heading_forms(plain: str) -> tuple[str, str]:
    collapsed = collapse_spaced_caps(plain)
    h = _fuzzy_lower(collapsed).rstrip(":;.- ")
    compact = re.sub(r"[^a-z0-9]", "", h)
    return h, compact


def plain_line(raw: str) -> str:
    line = raw.strip()
    line = re.sub(r"^#{1,6}\s+", "", line)
    line = re.sub(r"[*#_`]+", " ", line)
    line = BULLET_PREFIX_RE.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def _matches_heading_key(h: str, key: str) -> bool:
    if key == "project" and h not in {"project", "projects"}:
        return False
    if key == "project" and h in {"project", "projects"}:
        return h == key
    if h == key or h.startswith(key + " ") or h.startswith(key + ":"):
        return True
    if key in ENDSWITH_BLOCKLIST:
        return False
    if h.endswith(" " + key) or (len(key) > 4 and h.endswith(key)):
        return True
    return False


def resolve_section_type(plain: str) -> str:
    h, compact = _heading_forms(plain)
    for key in HEADING_KEYS:
        if _matches_heading_key(h, key):
            if key == "project" and not plain.isupper():
                continue
            return FULL_MAP[key]
    if COMPACT_SECTION_RE.match(compact):
        return "EXPERIENCE"
    if re.match(r"^uiux(?:design(?:ing)?)?$", compact):
        return "PROJECTS"
    if plain.isupper() and len(plain.split()) <= 4:
        for key in HEADING_KEYS:
            if h == key:
                return FULL_MAP[key]
    return "OTHER"


def _is_experience_entry_header(plain: str) -> bool:
    return bool(EXP_ENTRY_RE.search(plain))


def _is_degree_entry_header(plain: str) -> bool:
    return bool(DEGREE_LINE_RE.match(plain))


def is_false_heading(plain: str, raw: str) -> bool:
    if not plain or len(plain) > 90:
        return True
    if CONTACT_FIELD_RE.match(plain) or FALSE_HEADING_RE.match(plain) or DATE_ONLY_RE.match(plain):
        return True
    if re.match(r"^[-—–]\s*", plain) and resolve_section_type(plain) == "OTHER":
        return True
    if _is_experience_entry_header(plain) or _is_degree_entry_header(plain):
        return False
    sec = resolve_section_type(plain)
    if sec == "OTHER" and JOB_TITLE_RE.search(plain) and not plain.isupper():
        if not _is_experience_entry_header(plain):
            return True
    if "|" in plain and "linkedin" not in plain.lower() and sec == "OTHER":
        if SKILL_PIPE_RE.match(plain) and len(plain) < 28:
            return False
        if _is_degree_entry_header(plain):
            return False
        return True
    if plain.lower() == "contact" and not raw.strip().isupper():
        return True
    if re.match(r"^\d", plain) and resolve_section_type(plain) == "OTHER":
        return True
    if re.search(r"\blanguages?\s+known\b", plain, re.I):
        return True
    if re.search(r"\bfresher\b", plain, re.I):
        return True
    if plain.lower() in {"final year project", "linkedin id"}:
        return True
    return False


def _is_known_heading(plain: str) -> bool:
    sec = resolve_section_type(plain)
    if sec != "OTHER":
        return True
    h, _ = _heading_forms(plain)
    if h in OTHER_SECTION_HEADINGS:
        return True
    return any(_matches_heading_key(h, k) for k in OTHER_SECTION_HEADINGS)


def _alpha_len(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


def _is_skill_subheading(raw: str, plain: str) -> bool:
    if not BULLET_PREFIX_RE.match(raw.strip()):
        return False
    if len(plain) > 28 or re.search(r"\b(and|the|with|from|using|worked|performed)\b", plain, re.I):
        return False
    if plain.endswith("|"):
        return bool(SKILL_PIPE_RE.match(plain))
    return len(plain.split()) <= 4 and plain[0].isupper()


def _line_style_is_heading(line: LineRecord, doc_median_font: float, plain: str) -> bool:
    words = plain.split()
    is_caps_short = plain.isupper() and len(words) <= 6
    h, _ = _heading_forms(plain)

    if line.is_bold:
        return True
    if line.font_median >= doc_median_font * 1.03:
        return True
    if is_caps_short and len(words) >= 2:
        return True
    if h in SINGLE_WORD_HEADINGS and plain.isupper():
        return True
    if plain.endswith(":") and len(plain) < 55 and resolve_section_type(plain) != "OTHER":
        return True
    return False


def predict_line_heading(
    line: LineRecord,
    doc_median_font: float,
    *,
    lines_since_skills: int | None = None,
    lines_since_work_history: int | None = None,
) -> tuple[bool, float]:
    """Return (is_heading, confidence) for a single line."""
    raw = line.text
    plain = plain_line(raw)
    plain = re.sub(r"^\d+%\s*", "", plain).strip()

    if _alpha_len(raw) > MAX_ALPHA_LEN or is_false_heading(plain, raw):
        return False, 0.95

    if lines_since_skills is not None and lines_since_skills <= 2 and _is_skill_subheading(raw, plain):
        return True, 0.78

    known = _is_known_heading(plain)
    style_ok = _line_style_is_heading(line, doc_median_font, plain)

    if known and style_ok:
        return True, 0.92
    if known and plain.isupper():
        return True, 0.75
    if (
        lines_since_work_history is not None
        and lines_since_work_history <= 2
        and _is_experience_entry_header(plain)
        and line.is_bold
    ):
        return True, 0.82
    return False, 0.6 if known else 0.3


def _section_anchor_distance(lines: list[LineRecord], idx: int, keywords: frozenset[str]) -> int | None:
    for back in range(1, 8):
        j = idx - back
        if j < 0:
            break
        h, _ = _heading_forms(plain_line(lines[j].text))
        if h in keywords or any(k in h for k in keywords):
            return back
    return None


def predict_heading_lines(tokens: list[dict[str, Any]]) -> set[tuple[int, int]]:
    """H0: heuristic-only heading line keys (parser lineIndex)."""
    lines = build_parser_lines(tokens)
    fonts = [float(t.get("fontSize", 11.0) or 11.0) for t in tokens]
    doc_median_font = sorted(fonts)[len(fonts) // 2] if fonts else 11.0

    skills_keys = frozenset({"skills", "technical skills", "core competencies", "areas of expertise"})
    work_keys = frozenset({"work history", "work experience", "professional experience", "experience", "employment"})

    pred: set[tuple[int, int]] = set()
    for i, line in enumerate(lines):
        since_skills = _section_anchor_distance(lines, i, skills_keys)
        since_work = _section_anchor_distance(lines, i, work_keys)
        is_head, conf = predict_line_heading(
            line,
            doc_median_font,
            lines_since_skills=since_skills,
            lines_since_work_history=since_work,
        )
        if is_head and conf >= 0.7:
            pred.add(line.key)
    return dedupe_adjacent_heading_keys(lines, pred)


def compute_fha(gt_lines: set[tuple[int, int]], pred_lines: set[tuple[int, int]]) -> float:
    tp = len(gt_lines & pred_lines)
    union = len(gt_lines | pred_lines)
    return (tp / union * 100.0) if union > 0 else 100.0


def precision_recall(
    gt_lines: set[tuple[int, int]], pred_lines: set[tuple[int, int]]
) -> tuple[float, float]:
    tp = len(gt_lines & pred_lines)
    prec = tp / len(pred_lines) if pred_lines else 0.0
    rec = tp / len(gt_lines) if gt_lines else 0.0
    return prec * 100.0, rec * 100.0
