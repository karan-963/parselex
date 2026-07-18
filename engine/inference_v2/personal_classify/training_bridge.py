"""Lazy imports from engine/parity/personal (parity only)."""

from __future__ import annotations

import importlib
import os
import sys

from inference_v2.training_path import prioritize_training_paths

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PERSONAL_DIR = os.path.join(_REPO_ROOT, "engine", "parity", "personal")
_EXPERIENCE_PHASE3_DIR = os.path.join(_REPO_ROOT, "engine", "parity", "experience", "phase3_segment_classification")

_HELPERS_CACHE: dict | None = None


def _ensure_paths() -> tuple[str, str]:
    prioritize_training_paths(_EXPERIENCE_PHASE3_DIR, _PERSONAL_DIR)
    return _PERSONAL_DIR, _EXPERIENCE_PHASE3_DIR


def _personal_config_path() -> str:
    return os.path.join(_PERSONAL_DIR, "config.py")


def _clear_stale_modules() -> None:
    stale_config = sys.modules.get("config")
    if stale_config is not None and getattr(stale_config, "__file__", "") != _personal_config_path():
        del sys.modules["config"]
    for name in (
        "dataset",
        "segment_heuristics",
        "segment_postprocess",
        "spatial_utils",
        "strategy",
        "generate_spacial_reports",
        "classifier_config",
    ):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_path = getattr(mod, "__file__", "") or ""
        if not mod_path.startswith(_PERSONAL_DIR) and not mod_path.startswith(_EXPERIENCE_PHASE3_DIR):
            del sys.modules[name]


def load_training_helpers() -> dict:
    """Return training parity helpers for segment build, spatial, and postprocess."""
    global _HELPERS_CACHE
    _ensure_paths()
    _clear_stale_modules()
    if _HELPERS_CACHE is not None:
        return _HELPERS_CACHE

    generate_spacial_reports = importlib.import_module("generate_spacial_reports")
    dataset = importlib.import_module("dataset")
    segment_heuristics = importlib.import_module("segment_heuristics")
    segment_postprocess = importlib.import_module("segment_postprocess")
    spatial_utils = importlib.import_module("spatial_utils")
    personal_config = importlib.import_module("config")

    _HELPERS_CACHE = {
        "clean_cid_tokens": generate_spacial_reports.clean_cid_tokens,
        "build_personal_segments": segment_heuristics.build_personal_segments,
        "derive_segment_label": segment_heuristics.derive_segment_label,
        "get_segment_class_label": dataset.get_segment_class_label,
        "extract_segment_spatial": spatial_utils.extract_segment_spatial,
        "post_process_segment_predictions": segment_postprocess.post_process_segment_predictions,
        "segment_labels_match": segment_postprocess.segment_labels_match,
        "LABEL_LIST": personal_config.LABEL_LIST,
    }
    return _HELPERS_CACHE
