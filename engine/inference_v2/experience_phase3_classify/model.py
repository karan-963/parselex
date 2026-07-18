"""Entry-block field classifier model — wraps ResumeChunkClassifier from training-engine."""

from __future__ import annotations

import os
import sys

_MODULE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.abspath(os.path.join(_MODULE, "..", "..", ".."))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from models.chunk_model import ResumeChunkClassifier  # noqa: E402

__all__ = ["ResumeChunkClassifier", "build_classifier"]


def build_classifier(*, num_labels: int, spatial_dim: int, model_name: str) -> ResumeChunkClassifier:
    return ResumeChunkClassifier(
        num_labels=num_labels,
        spatial_dim=spatial_dim,
        model_name=model_name,
    )
