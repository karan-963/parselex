"""Load project entry heads from MongoDB when available."""

from __future__ import annotations

import os
import sys

_ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from inference_v2.overlay_mongo_labels import load_mongo_entry_heads


def load_entry_head_lines(resume_id: str) -> set[tuple[int, int]]:
    return load_mongo_entry_heads(resume_id, "PROJECT")
