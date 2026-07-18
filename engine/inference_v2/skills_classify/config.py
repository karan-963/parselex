"""Skills token BIO classifier — inference config (training parity)."""

from __future__ import annotations

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPATIAL_DIM = 16
MAX_SEGS = 128
MAX_SEG_LEN = 32
GLU_HIDDEN_SIZE = 256

LABEL_LIST = ["O", "B-SKILL", "I-SKILL", "B-SKILL_TYPE", "I-SKILL_TYPE"]
LABEL2ID = {label: index for index, label in enumerate(LABEL_LIST)}
ID2LABEL = {index: label for index, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)
