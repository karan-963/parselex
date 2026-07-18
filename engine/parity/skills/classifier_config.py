"""Skills token classifier training config (MiniLM)."""

from __future__ import annotations

import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

BACKBONE_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RUN_ID = "skills_mini_v1"
FRESH_TRAIN = os.getenv("SKILLS_FRESH", "0") == "1"

CHECKPOINT_DIR = os.path.join(MODULE_DIR, "saved_models", "minilm")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
ROLLING_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")
ONNX_PATH = os.path.join(CHECKPOINT_DIR, "skills_segment_classifier.onnx")
REPORTS_DIR = os.path.join(MODULE_DIR, "reports", "minilm")

LEGACY_CHECKPOINT_PATH = os.path.join(MODULE_DIR, "saved_models", "skills_segment_classifier.pt")
LEGACY_REPORTS_DIR = os.path.join(MODULE_DIR, "reports")

SPATIAL_DIM = 16
EPOCHS = 25
BATCH_SIZE = 2
MIN_VAL_MACRO_F1 = 75.0
