"""Lazy imports from engine/parity/education/new_phase2_section_divider (parity only)."""

from __future__ import annotations

import importlib
import os
import sys

from inference_v2.training_path import prioritize_training_paths

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PHASE2_DIR = os.path.join(_REPO_ROOT, "engine", "parity", "education", "new_phase2_section_divider")

_HELPERS_CACHE: dict | None = None


def _ensure_phase2_path() -> str:
    prioritize_training_paths(_PHASE2_DIR)
    return _PHASE2_DIR


def _phase2_config_path() -> str:
    return os.path.join(_PHASE2_DIR, "config.py")


def _is_stale_config(mod: object) -> bool:
    return getattr(mod, "__file__", "") != _phase2_config_path()


def _clear_stale_phase2_modules() -> None:
    stale_config = sys.modules.get("config")
    if stale_config is not None and _is_stale_config(stale_config):
        del sys.modules["config"]
    for name in (
        "dataset",
        "spatial_segments",
        "education_spatial_builder",
        "education_boundary_postprocess",
        "education_line_utils",
        "education_report_helpers",
        "education_entry_heads",
        "boundary_config",
    ):
        mod = sys.modules.get(name)
        if mod is not None and not getattr(mod, "__file__", "").startswith(_PHASE2_DIR):
            del sys.modules[name]


def load_training_helpers():
    """Return training parity helpers for segment build, spatial, and postprocess."""
    global _HELPERS_CACHE
    _ensure_phase2_path()
    _clear_stale_phase2_modules()
    if _HELPERS_CACHE is not None:
        return _HELPERS_CACHE

    spatial_segments = importlib.import_module("spatial_segments")
    dataset = importlib.import_module("dataset")
    spatial_builder = importlib.import_module("education_spatial_builder")
    boundary_postprocess = importlib.import_module("education_boundary_postprocess")
    line_utils = importlib.import_module("education_line_utils")
    report_helpers = importlib.import_module("education_report_helpers")

    _HELPERS_CACHE = {
        "clean_cid_tokens": spatial_segments.clean_cid_tokens,
        "construct_sentences_by_appearance": spatial_segments.construct_sentences_by_appearance,
        "group_segments_by_line": dataset.group_segments_by_line,
        "is_education_segment": dataset.is_education_segment,
        "build_segment_spatial_features": spatial_builder.build_segment_spatial_features,
        "pad_spatial_features": spatial_builder.pad_spatial_features,
        "apply_full_boundary_pipeline": boundary_postprocess.apply_full_boundary_pipeline,
        "build_physical_line_text_map": line_utils.build_physical_line_text_map,
        "collect_education_line_coords": report_helpers.collect_education_line_coords,
    }
    return _HELPERS_CACHE
