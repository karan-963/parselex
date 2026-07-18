"""
config.py
=========
Central configuration for the Skills Token-Level Sequence Classification pipeline.

Defines:
  - LABEL_LIST  : 5-class BIO token-level label space
  - LABEL2ID / ID2LABEL : bidirectional mappings
  - NUM_LABELS  : derived count used to size the model head (5)
  - EXCLUDED_RESUMES : global 6-resume blacklist that must be applied consistently
                       across every dataset loading pass to prevent split contamination.
  - Training hyper-parameters
"""

import os

import classifier_config as _cfg

# ─── Label Space ────────────────────────────────────────────────────────────────
LABEL_LIST = ["O", "B-SKILL", "I-SKILL", "B-SKILL_TYPE", "I-SKILL_TYPE"]

LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL: dict[int, str] = {i: lbl for i, lbl in enumerate(LABEL_LIST)}
NUM_LABELS: int = len(LABEL_LIST)  # 5

# ─── Split Contamination Guard ───────────────────────────────────────────────────
EXCLUDED_RESUMES: set[str] = {
    "Shubham_Mishra_Resume_BA_PO",
    "rohan_shirsat_resume-1",
    "Harshada_Bairagi_CV",
    "SUMEET_KUMAR_CV_2026",
    "Amit_Kharade_2026",
    "Paresh_Gudhka_Profile",
}

# ─── MongoDB Connection ──────────────────────────────────────────────────────────
MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB: str = os.getenv("MONGO_DB", "resume-labeling")

# ─── Model & Tokenizer (MiniLM) ──────────────────────────────────────────────────
ENCODER_NAME: str = _cfg.BACKBONE_NAME
CHECKPOINT_PATH: str = _cfg.CHECKPOINT_PATH
REPORTS_DIR: str = _cfg.REPORTS_DIR

# ─── Segment Sequence Dimensions ─────────────────────────────────────────────────
MAX_SEGS: int = 128
MAX_SEG_LEN: int = 32
SPATIAL_DIM: int = _cfg.SPATIAL_DIM

# ─── Training Hyperparameters ─────────────────────────────────────────────────────
BATCH_SIZE: int = _cfg.BATCH_SIZE
TOTAL_EPOCHS: int = _cfg.EPOCHS
ENCODER_LR: float = 2e-5
HEAD_LR: float = 1e-3
WEIGHT_DECAY: float = 0.01
GRAD_CLIP: float = 1.0
ACCUMULATION_STEPS: int = 2

# [O, B-SKILL, I-SKILL, B-SKILL_TYPE, I-SKILL_TYPE]
CLASS_WEIGHTS: list[float] = [1.0, 4.0, 3.0, 5.0, 4.0]

GLU_HIDDEN_SIZE: int = 256


def is_skills_section_excluded(doc: dict) -> bool:
    if doc.get("trainingMeta", {}).get("split") == "excluded":
        return True
    return "skills" in doc.get("excludedSections", [])
