"""
config.py
=========
Central configuration for the Personal/Meta Segment Classification training pipeline.
"""

import os

import classifier_config as _cfg

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
NUM_LABELS: int = len(LABEL_LIST)

EXCLUDED_RESUMES: set[str] = {
    "Shubham_Mishra_Resume_BA_PO",
    "rohan_shirsat_resume-1",
    "Harshada_Bairagi_CV",
    "SUMEET_KUMAR_CV_2026",
    "Amit_Kharade_2026",
    "Paresh_Gudhka_Profile",
}

MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB: str = os.getenv("MONGO_DB", "resume-labeling")

ENCODER_NAME: str = _cfg.BACKBONE_NAME
CHECKPOINT_PATH: str = _cfg.CHECKPOINT_PATH
REPORTS_DIR: str = _cfg.REPORTS_DIR

MAX_SEGS: int = 128
MAX_SEG_LEN: int = 32
SPATIAL_DIM: int = _cfg.SPATIAL_DIM

BATCH_SIZE: int = _cfg.BATCH_SIZE
TOTAL_EPOCHS: int = _cfg.EPOCHS
ENCODER_LR: float = 2e-5
HEAD_LR: float = 1e-3
WEIGHT_DECAY: float = 0.01
GRAD_CLIP: float = 1.0
ACCUMULATION_STEPS: int = 2

GLU_HIDDEN_SIZE: int = 256


def is_personal_section_excluded(doc: dict) -> bool:
    if doc.get("trainingMeta", {}).get("split") == "excluded":
        return True
    return "personal" in doc.get("excludedSections", [])
