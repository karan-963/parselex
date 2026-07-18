"""Lazy imports from engine/parity/skills (parity only)."""

from __future__ import annotations

import importlib
import os
import sys

from inference_v2.training_path import prioritize_training_paths

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SKILLS_DIR = os.path.join(_REPO_ROOT, "engine", "parity", "skills")
_EXPERIENCE_PHASE3_DIR = os.path.join(_REPO_ROOT, "engine", "parity", "experience", "phase3_segment_classification")

_HELPERS_CACHE: dict | None = None


def _ensure_paths() -> tuple[str, str]:
    prioritize_training_paths(_EXPERIENCE_PHASE3_DIR, _SKILLS_DIR)
    return _SKILLS_DIR, _EXPERIENCE_PHASE3_DIR


def _skills_config_path() -> str:
    return os.path.join(_SKILLS_DIR, "config.py")


def _clear_stale_modules() -> None:
    stale_config = sys.modules.get("config")
    if stale_config is not None and getattr(stale_config, "__file__", "") != _skills_config_path():
        del sys.modules["config"]
    for name in ("data", "dataset", "spatial_utils", "strategy", "generate_spacial_reports"):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_path = getattr(mod, "__file__", "") or ""
        if not mod_path.startswith(_SKILLS_DIR) and not mod_path.startswith(_EXPERIENCE_PHASE3_DIR):
            del sys.modules[name]


def load_training_helpers() -> dict:
    """Return training parity helpers for segment build and token label mapping."""
    global _HELPERS_CACHE
    _ensure_paths()
    _clear_stale_modules()
    if _HELPERS_CACHE is not None:
        return _HELPERS_CACHE

    generate_spacial_reports = importlib.import_module("generate_spacial_reports")
    dataset = importlib.import_module("dataset")
    spatial_utils = importlib.import_module("spatial_utils")
    data = importlib.import_module("data")
    strategy = importlib.import_module("strategy")
    _HELPERS_CACHE = {
        "clean_cid_tokens": generate_spacial_reports.clean_cid_tokens,
        "construct_sentences_by_appearance": generate_spacial_reports.construct_sentences_by_appearance,
        "is_skills_segment": dataset.is_skills_segment,
        "is_heading_segment": dataset.is_heading_segment,
        "split_skills_segment": dataset.split_skills_segment,
        "extract_segment_spatial": spatial_utils.extract_segment_spatial,
        "map_label_to_5class": data.map_label_to_5class,
        "skills_5class_map": data.SKILLS_5CLASS_MAP,
        "post_process_predictions": strategy.post_process_predictions,
    }
    return _HELPERS_CACHE
