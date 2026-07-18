"""Token-level regex refinement for atomic contact fields (phone / email / URL).

The segment classifier assigns a single label per atomic segment. When a segment
mixes a phone and an email — e.g. ``+91-9096219369 # yuvrajkhade2005@gmail.com``
where ``#`` is not a hard separator — the whole span inherits one label and the
phone is mislabeled as EMAIL. This pass re-labels individual tokens whose text is
an unambiguous contact atom, leaving multi-token fields (NAME / POSITION /
LOCATION) to the segment model.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL_RE = re.compile(
    r"https?://|www\.|"
    r"\b[\w-]+\.(?:com|net|org|io|dev|me|co|in|ai|app|xyz|tech|info|biz|"
    r"link|site|page|portfolio|cc|gg|sh)\b",
    re.IGNORECASE,
)

# Connector punctuation that should never carry a field label on its own.
_RESET_PUNCT = frozenset({"#", "§", "|", "—", "–", "•", "·", "▪", "■", "/", ";", "*", "~"})

# URL scheme fragments that PDF extraction sometimes splits off as separate
# tokens (e.g. "https" / ":" / "//github.com/x"). None of these match a
# contact pattern on their own, so they're buffered and folded into the link
# entity that follows, instead of being left unlabeled and dropped from the URL.
_SCHEME_TOKENS = frozenset({"http", "https", "http:", "https:", ":"})
_LINK_ENTITIES = frozenset({"GITHUB", "LINKEDIN", "OTHER_LINK"})


def _phone_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _is_phone(text: str) -> bool:
    """Phone-shaped and long enough to not collide with dates/pincodes.

    Requires either an explicit ``+`` country prefix (7-15 digits) or at least
    10 digits, so 6-digit pincodes and 8-digit dd-mm-yyyy dates are excluded.
    """
    raw = text.strip()
    digits = _phone_digits(raw)
    if not (7 <= len(digits) <= 15):
        return False
    if not re.fullmatch(r"\+?[\d\s\-().]+", raw):
        return False
    return raw.startswith("+") or len(digits) >= 10


def classify_contact_token(text: str) -> str | None:
    """Return the entity name for an unambiguous contact token, else None.

    Returns ``"O"`` for connector punctuation, an entity string (``EMAIL``,
    ``PHONE``, ``LINKEDIN``, ``GITHUB``, ``OTHER_LINK``) for a contact atom, or
    ``None`` to keep the token's existing segment label.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    if raw in _RESET_PUNCT:
        return "O"
    low = raw.lower()
    if _EMAIL_RE.search(raw):
        return "EMAIL"
    if "linkedin.com" in low:
        return "LINKEDIN"
    if "github.com" in low:
        return "GITHUB"
    if _is_phone(raw):
        return "PHONE"
    if _URL_RE.search(low):
        return "OTHER_LINK"
    return None


def _entity_of(bio_label: str | None) -> str | None:
    if not bio_label or bio_label == "O" or "-" not in bio_label:
        return None
    return bio_label.split("-", 1)[1]


def _set_bio(token: dict, bio: str) -> None:
    token["bioLabel"] = bio
    token["bio_label"] = bio


def refine_personal_token_labels(tokens: list[dict], indices: list[int]) -> None:
    """Override individual contact tokens in place, preserving BIO continuity."""
    prev_entity: str | None = None
    scheme_buffer: list[int] = []

    for idx in indices:
        token = tokens[idx]
        raw = (token.get("token", "") or "").strip().lower()
        override = classify_contact_token(token.get("token", ""))

        if override is None and raw in _SCHEME_TOKENS:
            scheme_buffer.append(idx)
            continue

        if override in _LINK_ENTITIES and scheme_buffer:
            # Fold the buffered scheme prefix ("https", ":") into this link
            # entity so the reassembled URL is complete.
            first_idx = scheme_buffer[0]
            _set_bio(tokens[first_idx], f"B-{override}")
            for buf_idx in scheme_buffer[1:]:
                _set_bio(tokens[buf_idx], f"I-{override}")
            scheme_buffer = []
            _set_bio(token, f"I-{override}")
            prev_entity = override
            continue

        if scheme_buffer:
            for buf_idx in scheme_buffer:
                _set_bio(tokens[buf_idx], "O")
            scheme_buffer = []

        if override is None:
            prev_entity = _entity_of(token.get("bioLabel"))
            continue

        if override == "O":
            _set_bio(token, "O")
            prev_entity = None
            continue

        bio = f"B-{override}" if prev_entity != override else f"I-{override}"
        _set_bio(token, bio)
        prev_entity = override

    if scheme_buffer:
        for buf_idx in scheme_buffer:
            _set_bio(tokens[buf_idx], "O")
