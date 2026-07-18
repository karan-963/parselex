"""Skills eval helpers — heading filter and alphanumeric gate."""

from __future__ import annotations

import re


def coord_key(token: dict) -> tuple[int, float, float]:
    return (
        int(token.get("page", 0)),
        round(float(token.get("x0", 0.0)), 2),
        round(float(token.get("y0", 0.0)), 2),
    )


def is_evaluable_token(token_text: str) -> bool:
    return bool(re.search(r"[a-zA-Z0-9]", token_text or ""))


def alnum_core(token_text: str) -> str:
    if not token_text:
        return ""
    stripped = re.sub(r"^[^a-zA-Z0-9]+", "", token_text.strip())
    return re.sub(r"[^a-zA-Z0-9]+$", "", stripped)


_HEADING_PATTERNS = {
    "skills", "technical skills", "skill summary", "technologies",
    "technology", "core competencies", "technical_skills", "competencies",
}


def is_heading_line(line_tokens: list[dict], token_text: str) -> bool:
    sorted_tokens = sorted(line_tokens, key=lambda token: float(token.get("x0", 0.0)))
    line_text = " ".join(token.get("token") or "" for token in sorted_tokens).strip()
    text_lower = line_text.lower()
    text_clean = re.sub(r"[^a-z\s]", "", text_lower).strip()
    if ":" in text_lower:
        return False
    if text_clean in _HEADING_PATTERNS:
        return True
    if token_text.strip().upper() in {"SKILLS", "TECHNICAL", "CODING", "LANGUAGE", "SUMMARY"}:
        return True
    for token in line_tokens:
        raw_bio = (token.get("bioLabel") or "").upper()
        section = (token.get("section") or "").upper()
        label = (token.get("label") or "").upper()
        if "HEADING" in raw_bio or "HEADING" in label or section == "HEADING":
            return True
    return False
