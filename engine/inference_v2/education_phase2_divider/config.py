"""Education training phase 2 — section divider config."""

from __future__ import annotations

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPATIAL_DIM = 19
NUM_LABELS = 3
MAX_EVAL_SEGS = 512
MAX_SEG_LEN = 32

ID2LABEL = {0: "O", 1: "B-EDU_START", 2: "I-EDU_START"}
