"""Build per-segment spatial tensors for education P2 (shared train/eval/infer)."""

from __future__ import annotations

from collections import Counter

from education_line_utils import build_physical_line_text_map
from education_spatial_features import extract_education_spatial, spatial_zero_vector


def education_font_stats(
    segments: list,
    *,
    is_education_segment,
) -> tuple[float, float, float]:
    sizes = [
        s["spatial"][0]
        for s in segments
        if is_education_segment(s) and s.get("spatial")
    ]
    if not sizes:
        sizes = [s["spatial"][0] for s in segments if s.get("spatial")] or [10.0]
    max_size = max(sizes)
    min_size = min(sizes)
    default_size = Counter(sizes).most_common(1)[0][0]
    return max_size, default_size, min_size


def build_segment_spatial_features(
    segments: list,
    *,
    is_education_segment,
    raw_tokens: list | None = None,
) -> list[list[float]]:
    line_text_by_coord = build_physical_line_text_map(segments, raw_tokens)
    max_size, default_size, min_size = education_font_stats(
        segments, is_education_segment=is_education_segment,
    )
    return [
        extract_education_spatial(
            segments[i],
            segments[i - 1] if i > 0 else None,
            max_size,
            default_size,
            min_size,
            segments=segments,
            seg_idx=i,
            line_text_by_coord=line_text_by_coord,
        )
        for i in range(len(segments))
    ]


def pad_spatial_features(
    spatial_features: list[list[float]],
    target_len: int,
) -> list[list[float]]:
    out = list(spatial_features[:target_len])
    while len(out) < target_len:
        out.append(spatial_zero_vector())
    return out
