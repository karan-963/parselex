"""Token classification boundary model wrapper."""
from __future__ import annotations
import os
import sys

_MODULE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.abspath(os.path.join(_MODULE, "..", "..", ".."))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from models.token_model import ResumeTokenClassifier  # noqa: E402

__all__ = ["ResumeTokenClassifier", "build_segmenter"]

def build_segmenter(*, num_labels: int, spatial_dim: int, model_name: str, use_crf: bool = False) -> ResumeTokenClassifier:
    return ResumeTokenClassifier(
        num_labels=num_labels,
        spatial_dim=spatial_dim,
        model_name=model_name,
        use_crf=use_crf,
    )
