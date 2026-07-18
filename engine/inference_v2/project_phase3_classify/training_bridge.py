"""Lazy imports from engine/parity/project/phase3_segment_classification (parity only)."""

from __future__ import annotations

import importlib
import os
import sys

from inference_v2.training_path import prioritize_training_paths

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PHASE3_DIR = os.path.join(_REPO_ROOT, "engine", "parity", "project", "phase3_segment_classification")

_HELPERS_CACHE: dict | None = None


def _ensure_phase3_path() -> str:
    prioritize_training_paths(_PHASE3_DIR)
    return _PHASE3_DIR


def _phase3_config_path() -> str:
    return os.path.join(_PHASE3_DIR, "config.py")


def _is_stale_config(mod: object) -> bool:
    return getattr(mod, "__file__", "") != _phase3_config_path()


def _clear_stale_phase3_modules() -> None:
    """Drop cached top-level modules that shadow phase3 training imports."""
    stale_config = sys.modules.get("config")
    if stale_config is not None and _is_stale_config(stale_config):
        del sys.modules["config"]
    for name in (
        "dataset",
        "segment_postprocess",
        "segment_label_rules",
        "segment_context",
        "generate_spacial_reports",
        "classifier_config",
    ):
        mod = sys.modules.get(name)
        if mod is not None and not getattr(mod, "__file__", "").startswith(_PHASE3_DIR):
            del sys.modules[name]


def load_training_helpers():
    """Return training parity helpers (segment build, spatial, postprocess)."""
    global _HELPERS_CACHE
    _ensure_phase3_path()
    _clear_stale_phase3_modules()
    if _HELPERS_CACHE is not None:
        return _HELPERS_CACHE

    generate_spacial_reports = importlib.import_module("generate_spacial_reports")
    dataset = importlib.import_module("dataset")
    segment_postprocess = importlib.import_module("segment_postprocess")

    _HELPERS_CACHE = {
        "clean_cid_tokens": generate_spacial_reports.clean_cid_tokens,
        "construct_sentences_by_appearance": generate_spacial_reports.construct_sentences_by_appearance,
        "split_hyphenated_segments": dataset.split_hyphenated_segments,
        "is_project_segment": dataset.is_project_segment,
        "get_segment_class_label": dataset.get_segment_class_label,
        "build_spatial_feature_matrix": dataset.build_spatial_feature_matrix,
        "postprocess_segment_predictions": segment_postprocess.postprocess_segment_predictions,
    }
    return _HELPERS_CACHE
