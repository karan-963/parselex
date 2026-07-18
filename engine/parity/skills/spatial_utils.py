"""Segment spatial features aligned with combined contract (_normalize_spatial 20D → 16D slice)."""

from __future__ import annotations

import importlib.util
import os

_ENGINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BASE_DATASET_PATH = os.path.join(_ENGINE_DIR, "data", "base_dataset.py")

_spec = importlib.util.spec_from_file_location("te_base_dataset", _BASE_DATASET_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
_normalize_spatial = _mod._normalize_spatial


def extract_segment_spatial(
    segment: dict,
    all_tokens: list[dict] | None = None,
    spatial_dim: int = 16,
) -> list[float]:
    """Mean-pool per-token normalized spatial vectors for one visual segment."""
    seg_tokens = segment.get("tokens") or []
    if not seg_tokens:
        return [0.0] * spatial_dim

    matrix = _normalize_spatial(seg_tokens, all_tokens=all_tokens, augment=False)
    if not matrix:
        return [0.0] * spatial_dim

    pooled = [0.0] * spatial_dim
    n = len(matrix)
    for row in matrix:
        for i in range(min(spatial_dim, len(row))):
            pooled[i] += float(row[i])
    return [v / n for v in pooled]


def extract_12d_spatial(segment: dict) -> list[float]:
    """Legacy alias — first 12 dims of contract spatial (deprecated)."""
    return extract_segment_spatial(segment, spatial_dim=12)[:12]
