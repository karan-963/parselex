"""Personal section — atomic segment field classifier (step 15)."""

from __future__ import annotations

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPATIAL_DIM = 24
MAX_SEGS = 128
MAX_SEG_LEN = 32
GLU_HIDDEN_SIZE = 256

LABEL_LIST = [
    "O",
    "B-NAME",
    "I-NAME",
    "B-PHONE",
    "I-PHONE",
    "B-EMAIL",
    "I-EMAIL",
    "B-LOCATION",
    "I-LOCATION",
    "B-POSITION",
    "I-POSITION",
    "B-LINKEDIN",
    "I-LINKEDIN",
    "B-GITHUB",
    "B-OTHER_LINK",
    "I-OTHER_LINK",
    "B-HEADING",
    "I-HEADING",
    "B-DOB",
]

LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL: dict[int, str] = {i: lbl for i, lbl in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

FINAL_LABEL_LIST = list(LABEL_LIST)
