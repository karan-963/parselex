"""Resumes with trainingMeta.split == 'excluded' must never enter train/val or metrics."""

from __future__ import annotations

import json
import os
from typing import Iterable

from . import config

MANIFEST_PATH = os.path.join(config.INPUT_DATA_DIR, "excluded_resume_ids.json")


def load_excluded_ids() -> set[str]:
    """Load cached excluded resume IDs (written by prepare_token_data.py)."""
    if not os.path.isfile(MANIFEST_PATH):
        return set()
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("resumeIds", []))


def is_excluded(resume_id: str, excluded: set[str] | None = None) -> bool:
    ids = excluded if excluded is not None else load_excluded_ids()
    return resume_id in ids


def list_active_json_files(data_dir: str, excluded: set[str] | None = None) -> list[str]:
    """JSON files in data_dir minus excluded resumes."""
    ids = excluded if excluded is not None else load_excluded_ids()
    files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".json")
    )
    return [f for f in files if os.path.splitext(os.path.basename(f))[0] not in ids]


def filter_doc_ids(doc_ids: Iterable[str], excluded: set[str] | None = None) -> list[str]:
    ids = excluded if excluded is not None else load_excluded_ids()
    return [d for d in doc_ids if d not in ids]


def purge_stale_per_resume_reports(per_resume_dir: str, active_doc_ids: set[str]) -> int:
    """Remove per-resume markdown for resumes no longer in the active eval set."""
    if not os.path.isdir(per_resume_dir):
        return 0
    removed = 0
    for name in os.listdir(per_resume_dir):
        if not name.endswith(".md"):
            continue
        doc_id = name[:-3]
        if doc_id not in active_doc_ids:
            os.remove(os.path.join(per_resume_dir, name))
            removed += 1
    return removed
