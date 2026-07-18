"""Path resolution for Inference V2 pipeline."""

from __future__ import annotations

import os
from typing import Literal

_ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(_ENGINE_DIR, ".."))

RUNS_DIR = os.path.join(_ENGINE_DIR, "inference_runs")
DEFAULT_PDF = os.path.join(REPO_ROOT, "full-database", "Karan.pdf")
EXTRACT_TOKENS = os.path.join(REPO_ROOT, "archive", "full_pipeline", "extract_tokens.py")

TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

StepStatus = Literal["LOCKED", "PENDING"]

# Master pipeline catalog — training phase order (1 → 2 → 3) within each section track.
# Artifact step numbers match training phase order per section (e.g. experience 8→9→10).
PIPELINE_STEPS: list[dict[str, str | int | None]] = [
    # Global
    {"inf_step": 1, "artifact": "1_extracted_tokens.json", "section": "global", "training_phase": None, "task": "PDF token extraction", "module": "extract.py", "status": "LOCKED"},
    {"inf_step": 2, "artifact": "2_section_headings.json", "section": "section", "training_phase": 1, "task": "Heading detection", "module": "section_p1/", "status": "LOCKED"},
    {"inf_step": 3, "artifact": "3_section_labels.json", "section": "section", "training_phase": 2, "task": "Section assignment", "module": "section_p2/", "status": "LOCKED"},
    # Education — phase 1 → 2 → 3
    {"inf_step": 4, "artifact": "4_education_segments.json", "section": "education", "training_phase": 1, "task": "Token segmentation", "module": "education_phase1_segment/", "status": "LOCKED"},
    {"inf_step": 5, "artifact": "5_education_boundaries.json", "section": "education", "training_phase": 2, "task": "Section divider", "module": "education_phase2_divider/", "status": "LOCKED"},
    {"inf_step": 6, "artifact": "6_education_fields.json", "section": "education", "training_phase": 3, "task": "Segment classification", "module": "education_phase3_classify/", "status": "LOCKED"},
    # Skills — direct classification
    {"inf_step": 7, "artifact": "7_skills_fields.json", "section": "skills", "training_phase": None, "task": "Direct classification", "module": "skills_classify/", "status": "LOCKED"},
    # Experience — phase 1 → 2 → 3 (artifact steps 8, 9, 10)
    {"inf_step": 8, "artifact": "8_experience_segments.json", "section": "experience", "training_phase": 1, "task": "Token segmentation", "module": "experience_phase1_segment/", "status": "LOCKED"},
    {"inf_step": 9, "artifact": "9_experience_boundaries.json", "section": "experience", "training_phase": 2, "task": "Section divider", "module": "experience_phase2_divider/", "status": "LOCKED"},
    {"inf_step": 10, "artifact": "10_experience_classification.json", "section": "experience", "training_phase": 3, "task": "Segment classification", "module": "experience_phase3_classify/", "status": "LOCKED"},
    # Project — phase 1 → 2 → 3 (artifact steps 11, 12, 13)
    {"inf_step": 11, "artifact": "11_project_segments.json", "section": "project", "training_phase": 1, "task": "Token segmentation", "module": "project_phase1_segment/", "status": "LOCKED"},
    {"inf_step": 12, "artifact": "12_project_boundaries.json", "section": "project", "training_phase": 2, "task": "Section divider", "module": "project_phase2_divider/", "status": "LOCKED"},
    {"inf_step": 13, "artifact": "13_project_fields.json", "section": "project", "training_phase": 3, "task": "Segment classification", "module": "project_phase3_classify/", "status": "LOCKED"},
    # Personal — direct classification
    {"inf_step": 15, "artifact": "15_personal_fields.json", "section": "personal", "training_phase": None, "task": "Segment classification", "module": "personal_classify/", "status": "LOCKED"},
    # Finalize
    {"inf_step": 14, "artifact": "14_final_classified_tokens.json", "section": "global", "training_phase": None, "task": "Final merge", "module": "entities.py", "status": "LOCKED"},
]

# Stages written by the live orchestrator (implemented steps only).
STAGE_ARTIFACTS = [
    ("extract_tokens", "1_extracted_tokens.json"),
    ("section_phase1", "2_section_headings.json"),
    ("section_phase2", "3_section_labels.json"),
    ("education_phase2_divider", "5_education_boundaries.json"),
    ("education_phase1_segment", "4_education_segments.json"),
    ("education_phase3_classify", "6_education_fields.json"),
    ("skills_classify", "7_skills_fields.json"),
    ("experience_phase2_divider", "9_experience_boundaries.json"),
    ("experience_phase1_segment", "8_experience_segments.json"),
    ("experience_phase3_classify", "10_experience_classification.json"),
    ("project_phase2_divider", "12_project_boundaries.json"),
    ("project_phase1_segment", "11_project_segments.json"),
    ("project_phase3_classify", "13_project_fields.json"),
    ("personal_classify", "15_personal_fields.json"),
    ("finalize", "14_final_classified_tokens.json"),
    ("finalize", "structured.json"),
]
