"""Segment spatial features — 24D personal contract (_normalize_spatial 20D + lexical flags)."""

from __future__ import annotations

import importlib.util
import os
import re

_ENGINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BASE_DATASET_PATH = os.path.join(_ENGINE_DIR, "data", "base_dataset.py")

_spec = importlib.util.spec_from_file_location("te_base_dataset", _BASE_DATASET_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
_normalize_spatial = _mod._normalize_spatial


def _normalize_spatial_personal(
    tokens: list[dict],
    all_tokens: list[dict] | None = None,
) -> list[list[float]]:
    base = _normalize_spatial(tokens, all_tokens=all_tokens, augment=False)
    extended: list[list[float]] = []
    for idx, t in enumerate(tokens):
        text = t.get("token", "")
        text_low = text.lower()
        embedded_url = (
            str(t.get("embeddedUrl") or t.get("hyperlink") or t.get("href") or "")
        ).lower().strip()
        is_email = 1.0 if (
            ("@" in text_low and "." in text_low)
            or "mailto:" in embedded_url
            or "mailto%" in embedded_url
        ) else 0.0
        clean_phone = re.sub(r"[\s\-\(\)\+]", "", text)
        is_phone = 1.0 if (clean_phone.isdigit() and 7 <= len(clean_phone) <= 15) else 0.0
        is_linkedin = 1.0 if ("linkedin.com" in text_low or "linkedin.com" in embedded_url) else 0.0
        is_github = 1.0 if ("github.com" in text_low or "github.com" in embedded_url) else 0.0
        extended.append(base[idx] + [is_email, is_phone, is_linkedin, is_github])
    return extended


def extract_segment_spatial(
    segment: dict,
    all_tokens: list[dict] | None = None,
    spatial_dim: int = 24,
) -> list[float]:
    """Mean-pool per-token 24D personal spatial vectors for one visual segment."""
    seg_tokens = segment.get("tokens") or []
    if not seg_tokens:
        return [0.0] * spatial_dim

    matrix = _normalize_spatial_personal(seg_tokens, all_tokens=all_tokens)
    if not matrix:
        return [0.0] * spatial_dim

    pooled = [0.0] * spatial_dim
    n = len(matrix)
    for row in matrix:
        for i in range(min(spatial_dim, len(row))):
            pooled[i] += float(row[i])
    return [v / n for v in pooled]


def extract_12d_spatial(segment: dict) -> list[float]:
    """Legacy alias — first 12 dims (deprecated)."""
    return extract_segment_spatial(segment, spatial_dim=12)[:12]
