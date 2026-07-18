"""Experience Phase 2 Config — entry-block field classifier (ROLE/COMP/DATE/DESC)."""

from __future__ import annotations

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPATIAL_DIM = 12          # spatial_proj.mlp.0.weight: [128, 12]
NUM_LABELS = 4            # DESC=0, ROLE=1, COMP=2, DATE=3
MAX_LENGTH = 128

# Canonical label list must match training order
LABEL_LIST = ["DESC", "ROLE", "COMP", "DATE"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABEL_LIST)}

# Sentinel for first block in an entry (warm-start: index = NUM_LABELS)
PREV_LABEL_SENTINEL = NUM_LABELS

# BIO tag → macro class (mirrors phase3_segment_classification/config.py)
BIO_TO_CLASS: dict[str, str] = {
    "B-ROLE":     "ROLE",  "I-ROLE":     "ROLE",
    "B-COMP":     "COMP",  "I-COMP":     "COMP",
    "B-COMP_LOC": "COMP",  "I-COMP_LOC": "COMP",
    "B-SDATE":    "DATE",  "I-SDATE":    "DATE",
    "B-EDATE":    "DATE",  "I-EDATE":    "DATE",
    "B-DESC":     "DESC",  "I-DESC":     "DESC",
    "B-ROLEMETA": "ROLE",  "I-ROLEMETA": "ROLE",
}

DELIMITERS = {"|", "•", ","}
