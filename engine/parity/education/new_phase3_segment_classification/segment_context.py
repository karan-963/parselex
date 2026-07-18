"""Education-block context features and hard-negative detection for phase 3 training."""

from __future__ import annotations

import re

from config import LABEL2ID

_BULLET_PREFIX = re.compile(r"^[•\-\*●·▪■◦✓✔\uf0b7\uf0a7\s]+")
_INST_KEYWORD = re.compile(
    r"\b(?:university|college|institute|institution|school|academy|polytechnic|"
    r"campus|iit|nit|bits|vit|srm|amity|symbiosis)\b",
    re.IGNORECASE,
)
_DEGREE_ABBR = re.compile(
    r"\b(?:b\.?\s*tech|b\.?\s*e\.?|m\.?\s*tech|m\.?\s*sc|m\.?\s*ba|mba|bca|mca|"
    r"b\.?\s*com|m\.?\s*com|ph\.?\s*d|b\.?\s*sc|diploma|bachelor|master|"
    r"post\s*graduate|under\s*graduate|pgdm|b\.?\s*a\.?)\b",
    re.IGNORECASE,
)
_GPA_LINE = re.compile(r"\b(?:cgpa|gpa|percentage|percent|marks?|score)\b", re.IGNORECASE)
_LOC_LINE = re.compile(r"\b(?:city|state|country|pin\s*code|pincode)\b", re.IGNORECASE)
_YEAR_RANGE = re.compile(r"\b(?:19|20)\d{2}\b")


def _strip_bullets(text: str) -> str:
    return _BULLET_PREFIX.sub("", text.strip()).strip()


def institution_likeness_score(text: str) -> float:
    """0–1 heuristic: how much segment text resembles an institution name."""
    clean = _strip_bullets(text)
    if not clean:
        return 0.0
    score = 0.0
    if _INST_KEYWORD.search(clean):
        score += 0.45
    if clean.isupper() and len(clean.split()) <= 8:
        score += 0.2
    if re.search(r"\b(?:of|at)\s+[A-Z]", clean):
        score += 0.15
    if len(clean.split()) <= 6 and not _DEGREE_ABBR.search(clean):
        score += 0.1
    return min(score, 1.0)


def degree_likeness_score(text: str) -> float:
    """0–1 heuristic: how much segment text resembles a degree line."""
    clean = _strip_bullets(text)
    if not clean:
        return 0.0
    score = 0.0
    if _DEGREE_ABBR.search(clean):
        score += 0.5
    if re.search(r"\b(?:honours?|honors?|specialization|major|minor)\b", clean, re.IGNORECASE):
        score += 0.25
    if "|" in clean and _DEGREE_ABBR.search(clean):
        score += 0.15
    return min(score, 1.0)


def is_institution_like_desc_hard_negative(text: str) -> bool:
    """True when DESCRIPTION text resembles an institution (primary FP driver)."""
    clean = _strip_bullets(text)
    if not clean or len(clean) < 3:
        return False
    if _GPA_LINE.search(clean) or _LOC_LINE.search(clean):
        return False
    if institution_likeness_score(clean) >= 0.35:
        if degree_likeness_score(clean) >= 0.5 and institution_likeness_score(clean) < 0.5:
            return False
        return True
    return False


def is_gpa_or_location_hard_negative(text: str) -> bool:
    clean = _strip_bullets(text)
    return bool(_GPA_LINE.search(clean) or _LOC_LINE.search(clean))


def enrich_spatial_with_education_context(
    education_indices: list[int],
    segments: list[dict],
    spatial_features: list[list[float]],
) -> list[list[float]]:
    """Inject block context + institution/degree likeness into spatial dims 13–15."""
    if not education_indices:
        return spatial_features

    total = max(len(education_indices), 1)
    since_header = 99

    for rank, seg_i in enumerate(education_indices):
        seg = segments[seg_i]
        text = seg.get("text", "")
        tier = float(seg["spatial"][8]) if seg.get("spatial") else 0.0
        bold = float(seg["spatial"][1]) if seg.get("spatial") else 0.0
        is_header_like = tier <= 1.0 or bold >= 1.0

        if is_header_like:
            since_header = 0
        else:
            since_header += 1

        feat = list(spatial_features[seg_i])
        gap = feat[13] if len(feat) > 13 else 0.0
        inst_score = institution_likeness_score(text)
        deg_score = degree_likeness_score(text)
        title_score = max(inst_score, deg_score * 0.85)
        feat[13] = 0.6 * gap + 0.4 * title_score
        feat[14] = rank / max(total - 1, 1)
        body_depth = min(since_header, 10) / 10.0
        feat[15] = max(body_depth, inst_score * (0.3 if rank > 0 else 0.0))
        spatial_features[seg_i] = feat

    return spatial_features


def build_sample_weights(labels: list[int], seg_texts: list[str]) -> list[float]:
    desc_id = LABEL2ID["DESCRIPTION"]
    inst_id = LABEL2ID["INSTITUTION"]
    deg_id = LABEL2ID["DEGREE"]
    weights: list[float] = []
    for lbl, text in zip(labels, seg_texts):
        if lbl == -100:
            weights.append(0.0)
        elif lbl == desc_id and is_institution_like_desc_hard_negative(text):
            weights.append(5.0)
        elif lbl == desc_id:
            weights.append(1.5)
        elif lbl == inst_id:
            weights.append(1.3)
        elif lbl == deg_id:
            weights.append(1.2)
        else:
            weights.append(1.0)
    return weights


def build_hard_negative_mask(labels: list[int], seg_texts: list[str]) -> list[float]:
    desc_id = LABEL2ID["DESCRIPTION"]
    mask: list[float] = []
    for lbl, text in zip(labels, seg_texts):
        if lbl == desc_id and is_institution_like_desc_hard_negative(text):
            mask.append(1.0)
        else:
            mask.append(0.0)
    return mask
