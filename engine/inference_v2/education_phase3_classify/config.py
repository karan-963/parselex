"""Education phase 3 — segment field classifier (INSTITUTION / DEGREE / DATE / DESCRIPTION)."""

from __future__ import annotations

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPATIAL_DIM = 16
MAX_SEGS = 128
MAX_SEG_LEN = 32

MODEL_LABEL_LIST = ["DESCRIPTION", "INSTITUTION", "DEGREE", "DATE"]
MODEL_LABEL2ID = {lbl: i for i, lbl in enumerate(MODEL_LABEL_LIST)}
MODEL_ID2LABEL = {i: lbl for i, lbl in enumerate(MODEL_LABEL_LIST)}

FINAL_LABEL_LIST = list(MODEL_LABEL_LIST)

MACRO_TO_BIO_PREFIX: dict[str, str] = {
    "INSTITUTION": "INST",
    "DEGREE": "DEG",
    "DATE": "SDATE",
}
