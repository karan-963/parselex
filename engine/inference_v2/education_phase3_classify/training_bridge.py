"""Lazy imports from engine/parity/education/new_phase3_segment_classification (parity only)."""

from __future__ import annotations

import importlib
import os
import sys

from inference_v2.training_path import prioritize_training_paths

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PHASE3_DIR = os.path.join(_REPO_ROOT, "engine", "parity", "education", "new_phase3_segment_classification")
_EXPERIENCE_DIR = os.path.join(_REPO_ROOT, "engine", "parity", "experience")

_HELPERS_CACHE: dict | None = None


def _ensure_paths() -> tuple[str, str]:
    prioritize_training_paths(_EXPERIENCE_DIR, _PHASE3_DIR)
    return _PHASE3_DIR, _EXPERIENCE_DIR


def _phase3_config_path() -> str:
    return os.path.join(_PHASE3_DIR, "config.py")


def _is_stale_config(mod: object) -> bool:
    return getattr(mod, "__file__", "") != _phase3_config_path()


def _clear_stale_modules() -> None:
    stale_config = sys.modules.get("config")
    if stale_config is not None and _is_stale_config(stale_config):
        del sys.modules["config"]
    for name in (
        "dataset",
        "segment_postprocess",
        "segment_label_rules",
        "segment_context",
        "classifier_config",
        "generate_spacial_reports",
    ):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_path = getattr(mod, "__file__", "") or ""
        if not mod_path.startswith(_PHASE3_DIR) and not mod_path.startswith(_EXPERIENCE_DIR):
            del sys.modules[name]


def load_training_helpers():
    """Return training parity helpers for segment build, spatial, and postprocess."""
    global _HELPERS_CACHE
    _ensure_paths()
    _clear_stale_modules()
    if _HELPERS_CACHE is not None:
        return _HELPERS_CACHE

    generate_spacial_reports = importlib.import_module("generate_spacial_reports")
    dataset = importlib.import_module("dataset")
    segment_postprocess = importlib.import_module("segment_postprocess")

    _HELPERS_CACHE = {
        "clean_cid_tokens": generate_spacial_reports.clean_cid_tokens,
        "construct_sentences_by_appearance": generate_spacial_reports.construct_sentences_by_appearance,
        "is_education_segment": dataset.is_education_segment,
        "get_segment_class_label": dataset.get_segment_class_label,
        "build_spatial_feature_matrix": dataset.build_spatial_feature_matrix,
        "postprocess_segment_predictions": segment_postprocess.postprocess_segment_predictions,
    }
    return _HELPERS_CACHE
