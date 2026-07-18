"""Project-block context features and hard-negative detection for phase 3 training."""

from __future__ import annotations

import re

from config import LABEL2ID

_BULLET_PREFIX = re.compile(r"^[•\-\*●·▪■◦✓✔\uf0b7\uf0a7\s]+")
_GITHUB_PIPE = re.compile(r"\|\s*(?:github|gitlab|demo|live|link)\b", re.IGNORECASE)
_COLON_TITLE = re.compile(r"^[A-Z][\w\s\-&()]{2,80}\s*:")
_DOMAIN_LABEL = re.compile(r"^domain\s*:", re.IGNORECASE)
_COMPANY_SUFFIX = re.compile(
    r"\b(?:limited|ltd\.?|inc\.?|corp\.?|pvt\.?|llc|technologies|solutions)\s*\.?\s*$",
    re.IGNORECASE,
)
_PROJECT_WORD = re.compile(r"\bproject\b", re.IGNORECASE)
_NARRATIVE_VERB = re.compile(
    r"\b(?:designed|developed|built|implemented|using|through|we\s|i\s|created|based)\b",
    re.IGNORECASE,
)
_CODE_LINK = re.compile(r"^code\s*link", re.IGNORECASE)
_LIVE_DEMO = re.compile(r"live\s*demo", re.IGNORECASE)
_PIPE_TECH_STACK = re.compile(
    r"\|\s*(?:MERN|MEAN|JWT|React|Spring|Socket|Redux|Razorpay|Cloudinary|Hibernate|Hadoop|Kafka)",
    re.IGNORECASE,
)
_NAME_PIPE_LIST = re.compile(r"^[\w\s\-&()]+\s*\|\s*[\w\s,.\-+()]+", re.IGNORECASE)
_TOPIC_LIST = re.compile(r"^(?:analysis|classification|detection|recognition)\s*,", re.IGNORECASE)
_API_LINE = re.compile(r"\bAPI\.?\s*$", re.IGNORECASE)
_PAREN_PREFIX = re.compile(r"^\(\s*\)\s+")
_NUMBERED_TITLE = re.compile(r"^\d+\.\s+\w", re.IGNORECASE)


def _strip_bullets(text: str) -> str:
    return _BULLET_PREFIX.sub("", text.strip()).strip()


def title_likeness_score(text: str) -> float:
    """0–1 heuristic: how much segment text resembles a project title."""
    clean = _strip_bullets(text)
    if not clean:
        return 0.0
    score = 0.0
    if _GITHUB_PIPE.search(clean):
        score += 0.35
    if _CODE_LINK.search(clean):
        score += 0.45
    if _LIVE_DEMO.search(clean):
        score += 0.35
    if _PIPE_TECH_STACK.search(clean):
        score += 0.4
    if _NAME_PIPE_LIST.match(clean):
        score += 0.35
    if _COLON_TITLE.match(clean) and len(clean.split()) <= 14:
        score += 0.3
    if _DOMAIN_LABEL.search(clean):
        score += 0.35
    if _COMPANY_SUFFIX.search(clean) and len(clean.split()) <= 5:
        score += 0.3
    if _PROJECT_WORD.search(clean) and len(clean.split()) <= 12:
        score += 0.25
    if _TOPIC_LIST.search(clean):
        score += 0.25
    if _API_LINE.search(clean):
        score += 0.2
    if _PAREN_PREFIX.match(clean):
        score += 0.2
    if _NUMBERED_TITLE.match(clean):
        score += 0.2
    if clean.endswith("|") or clean.rstrip().endswith(","):
        score += 0.15
    return min(score, 1.0)


def is_title_like_desc_hard_negative(text: str) -> bool:
    """True when DESC text resembles a project title (primary FP driver)."""
    clean = _strip_bullets(text)
    if not clean or len(clean) < 3:
        return False
    if title_likeness_score(clean) >= 0.25:
        if _NARRATIVE_VERB.search(clean) and len(clean.split()) > 12:
            return False
        return True
    return False


def enrich_spatial_with_project_context(
    project_indices: list[int],
    segments: list[dict],
    spatial_features: list[list[float]],
) -> list[list[float]]:
    """Inject block context + title-likeness into spatial dims 13–15 (inference-safe)."""
    if not project_indices:
        return spatial_features

    total = max(len(project_indices), 1)
    since_header = 99

    for rank, seg_i in enumerate(project_indices):
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
        title_score = title_likeness_score(text)
        # Blend vertical gap with title-likeness (model learns optimal mix)
        feat[13] = 0.6 * gap + 0.4 * title_score
        feat[14] = rank / max(total - 1, 1)
        # Deep in block after headers → strongly favor DESC even if title-shaped
        body_depth = min(since_header, 10) / 10.0
        feat[15] = max(body_depth, title_score * (0.3 if rank > 0 else 0.0))
        spatial_features[seg_i] = feat

    return spatial_features


def build_sample_weights(labels: list[int], seg_texts: list[str]) -> list[float]:
    """Per-segment loss multipliers; aggressively upweight title-like DESC hard negatives."""
    desc_id = LABEL2ID["DESC"]
    proj_id = LABEL2ID["PROJECT_NAME"]
    weights: list[float] = []
    for lbl, text in zip(labels, seg_texts):
        if lbl == -100:
            weights.append(0.0)
        elif lbl == desc_id and is_title_like_desc_hard_negative(text):
            weights.append(5.0)
        elif lbl == desc_id:
            weights.append(1.5)
        elif lbl == proj_id:
            weights.append(1.2)
        else:
            weights.append(1.0)
    return weights


def build_hard_negative_mask(labels: list[int], seg_texts: list[str]) -> list[float]:
    """1.0 on title-like DESC segments that drive PROJECT_NAME false positives."""
    desc_id = LABEL2ID["DESC"]
    mask: list[float] = []
    for lbl, text in zip(labels, seg_texts):
        if lbl == desc_id and is_title_like_desc_hard_negative(text):
            mask.append(1.0)
        else:
            mask.append(0.0)
    return mask
