"""19D spatial features for education phase-2 boundary training."""

from __future__ import annotations

import boundary_config as bc
from education_layout_features import layout_context_dims

BULLETS = frozenset({
    "•", "*", "-", "▪", "◦", "■", "–", "—", "\uf0b7", "·", "✓", "✔", "\uf0a7", "●",
})


def spatial_dim() -> int:
    return 19


def spatial_zero_vector() -> list[float]:
    return [0.0] * spatial_dim()


def extract_16d_spatial(
    s: dict,
    prev_s: dict | None = None,
    max_size: float = 10.0,
    default_size: float = 10.0,
    min_size: float = 10.0,
) -> list[float]:
    fs = s["spatial"][0]
    bold = float(s["spatial"][1])
    page = float(s["spatial"][3])
    y0 = s["spatial"][4]
    x0 = s["spatial"][5]
    y1 = s["spatial"][6]
    x1 = s["spatial"][7]
    tier = float(s["spatial"][8])

    text = s["text"]
    is_all_caps = float(text.isupper() and len(text) > 2)
    has_bullet = float(any(text.startswith(b) for b in BULLETS))

    w = x1 - x0
    h = y1 - y0

    feat = [
        x0 / 612.0,
        y0 / 792.0,
        x1 / 612.0,
        y1 / 792.0,
        w / 612.0,
        h / 792.0,
        fs / 30.0,
        bold,
        is_all_caps,
        page / 10.0,
        tier / 3.0,
        has_bullet,
    ]

    if prev_s:
        fs_prev = prev_s["spatial"][0]
        bold_prev = float(prev_s["spatial"][1])
        page_prev = float(prev_s["spatial"][3])
        y1_prev = prev_s["spatial"][6]

        if page == page_prev:
            font_tier_delta = 1.0 if (fs < fs_prev or (bold_prev == 1.0 and bold == 0.0)) else 0.0
            visual_spacing_gap = (y0 - y1_prev) / 792.0
        else:
            font_tier_delta = 0.0
            visual_spacing_gap = 0.0
    else:
        font_tier_delta = 0.0
        visual_spacing_gap = 0.0

    feat.append(font_tier_delta)
    feat.append(visual_spacing_gap)

    feat_14 = 1.0 if (bold == 1.0 and abs(fs - max_size) < 1e-4) else 0.0

    diff_max = abs(fs - max_size)
    diff_default = abs(fs - default_size)
    diff_min = abs(fs - min_size)
    min_diff = min(diff_max, diff_default, diff_min)
    if min_diff == diff_max:
        feat_15 = 1.0
    elif min_diff == diff_default:
        feat_15 = 0.5
    else:
        feat_15 = 0.0

    feat.append(feat_14)
    feat.append(feat_15)
    return feat


def extract_education_spatial(
    s: dict,
    prev_s: dict | None = None,
    max_size: float = 10.0,
    default_size: float = 10.0,
    min_size: float = 10.0,
    *,
    segments: list | None = None,
    seg_idx: int | None = None,
    line_text_by_coord: dict[tuple[int, int], str] | None = None,
) -> list[float]:
    base = extract_16d_spatial(s, prev_s, max_size, default_size, min_size)
    if segments is not None and seg_idx is not None and line_text_by_coord is not None:
        layout = layout_context_dims(seg_idx, segments, line_text_by_coord)
    else:
        layout = [0.0, 0.0, 0.0]
    return base + layout


extract_16d_spatial_with_layout = extract_education_spatial

assert spatial_dim() == bc.SPATIAL_DIM, (
    f"SPATIAL_DIM={bc.SPATIAL_DIM} must match education_spatial_features ({spatial_dim()})"
)
