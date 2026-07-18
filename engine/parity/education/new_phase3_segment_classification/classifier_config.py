"""Education phase 3 segment classifier training config (MiniLM)."""

from __future__ import annotations

import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

BACKBONE_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RUN_ID = "edu_p3_mini_v1"

CHECKPOINT_DIR = os.path.join(MODULE_DIR, "saved_models", "minilm")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
ONNX_PATH = os.path.join(CHECKPOINT_DIR, "education_segment_classifier.onnx")
REPORTS_DIR = os.path.join(MODULE_DIR, "reports", "minilm")

LEGACY_CHECKPOINT_PATH = os.path.join(MODULE_DIR, "saved_models", "education_segment_classifier.pt")
LEGACY_REPORTS_DIR = os.path.join(MODULE_DIR, "reports")

SPATIAL_DIM = 16
EPOCHS = 25
BATCH_SIZE = 16
MIN_VAL_FBA = 75.0
