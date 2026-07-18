"""Segment-level post-processing for personal classification (atomic segments)."""

from __future__ import annotations

import re

from strategy import _is_phone_token

_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
_URL_RE = re.compile(
    r"https?://|www\.|"
    r"\b[\w-]+\.(?:com|net|org|io|dev|me|co|in|ai|app|xyz|tech|info|biz|"
    r"link|site|page|portfolio|cc|gg|sh)\b",
    re.IGNORECASE,
)
_STRUCTURAL_KEYWORDS = frozenset({
    "email", "e-mail", "mail", "mobile", "phone", "tel", "contact",
    "linkedin", "github", "website", "blog", "address",
})
_LINK_KEYWORDS = frozenset({
    "portfolio", "behance", "dribbble", "leetcode", "codepen", "medium",
})


def _looks_like_url(text: str) -> bool:
    """True when text resembles a web URL / domain (not an email)."""
    if not text or _EMAIL_RE.search(text):
        return False
    return bool(_URL_RE.search(text))


def normalize_segment_label(label: str) -> str:
    """Map I-{ENTITY} → B-{ENTITY}; leave O unchanged."""
    if label == "O" or not label:
        return "O"
    if label.startswith("I-"):
        return f"B-{label[2:]}"
    return label


def segment_labels_match(gt: str, pred: str) -> bool:
    """True when GT and prediction refer to the same atomic entity type."""
    return normalize_segment_label(gt) == normalize_segment_label(pred)


def _infer_label_from_text(text: str) -> str | None:
    """Infer canonical B-* label from segment text, or None to keep model output."""
    raw = text.strip()
    if not raw:
        return "O"
    low = raw.lower()

    if low in _STRUCTURAL_KEYWORDS:
        return "O"

    if _EMAIL_RE.search(low):
        return "B-EMAIL"
    if "linkedin.com" in low:
        return "B-LINKEDIN"
    if "github.com" in low:
        return "B-GITHUB"
    if _is_phone_token(raw) or (
        low.startswith("+") and any(c.isdigit() for c in raw)
    ):
        return "B-PHONE"
    if any(kw == low for kw in _LINK_KEYWORDS):
        return "B-OTHER_LINK"

    return None


def post_process_segment_prediction(segment: dict, pred_label: str) -> str:
    """
    Apply heuristic overrides on one atomic personal segment prediction.

    1. Normalize I-* → B-* (segment classifier emits one label per segment).
    2. Override from segment text when field type is unambiguous.
    3. Downgrade bare handles misclassified as B-GITHUB (no github.com in text).
    """
    text = (segment.get("text") or "").strip()
    label = normalize_segment_label(pred_label)

    inferred = _infer_label_from_text(text)
    if inferred is not None:
        return inferred

    # A URL the model tagged as LINKEDIN/GITHUB but that is neither platform
    # is a generic personal link (portfolio, custom domain, etc.).
    if label in ("B-LINKEDIN", "B-GITHUB") and _looks_like_url(text):
        low = text.lower()
        if "linkedin.com" not in low and "github.com" not in low:
            return "B-OTHER_LINK"

    return label


def post_process_segment_predictions(
    segments: list[dict],
    pred_labels: list[str],
) -> list[str]:
    """Post-process a parallel list of segment predictions."""
    out: list[str] = []
    for seg, lbl in zip(segments, pred_labels):
        out.append(post_process_segment_prediction(seg, lbl))
    return out
