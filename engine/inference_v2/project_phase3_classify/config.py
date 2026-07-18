"""Project phase 3 — segment field classifier (PROJECT_NAME / DATE / DESC → SDATE/EDATE)."""

from __future__ import annotations

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPATIAL_DIM = 16
MAX_SEGS = 128
MAX_SEG_LEN = 32

# Model output labels (training config.LABEL_LIST)
MODEL_LABEL_LIST = ["O", "PROJECT_NAME", "DATE", "DESC"]
MODEL_LABEL2ID = {lbl: i for i, lbl in enumerate(MODEL_LABEL_LIST)}
MODEL_ID2LABEL = {i: lbl for i, lbl in enumerate(MODEL_LABEL_LIST)}

# Resolved labels after date postprocess
FINAL_LABEL_LIST = ["PROJECT_NAME", "SDATE", "EDATE", "DESC"]

MACRO_TO_BIO_PREFIX: dict[str, str] = {
    "PROJECT_NAME": "PROJ_NAME",
    "SDATE": "SDATE",
    "EDATE": "EDATE",
    "DESC": "DESC",
}
