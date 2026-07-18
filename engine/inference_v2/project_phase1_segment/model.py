"""Phrase segmenter model wrapper."""
from __future__ import annotations

from .segmenter_model import PhraseSegmenterTransformer

__all__ = ["PhraseSegmenterTransformer", "build_segmenter"]


def build_segmenter(*, num_labels: int, spatial_dim: int, model_name: str, use_crf: bool = False) -> PhraseSegmenterTransformer:
    return PhraseSegmenterTransformer(
        num_labels=num_labels,
        spatial_dim=spatial_dim,
        model_name=model_name,
        use_crf=use_crf,
    )
