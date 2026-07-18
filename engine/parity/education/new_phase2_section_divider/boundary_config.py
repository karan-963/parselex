"""Education phase 2 entry-boundary divider training config (MiniLM)."""

from __future__ import annotations

import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

BACKBONE_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RUN_ID = "edu_p2_mini_v3_19d_layout"
FRESH_TRAIN = os.getenv("EDU_P2_FRESH", "1") == "1"

CHECKPOINT_DIR = os.path.join(MODULE_DIR, "saved_models", "minilm")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
ROLLING_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")
ONNX_PATH = os.path.join(CHECKPOINT_DIR, "education_section_divider.onnx")
REPORTS_DIR = os.path.join(MODULE_DIR, "reports", "minilm")

LEGACY_CHECKPOINT_PATH = os.path.join(MODULE_DIR, "saved_models", "education_section_divider.pt")
LEGACY_ONNX_PATH = os.path.join(MODULE_DIR, "saved_models", "education_section_divider.yx.onnx")
LEGACY_REPORTS_DIR = os.path.join(MODULE_DIR, "reports")

SPATIAL_DIM = 19
NUM_LABELS = 3
EPOCHS = 45
BATCH_SIZE = 16
MAX_EVAL_SEGS = 512

MIN_VAL_MODEL_FBA = 90.0
MIN_VAL_POSTPROCESS_FBA = 98.0

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "resume-labeling")
