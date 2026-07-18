"""Run folder management and manifest persistence."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any


def _runs_dir() -> str:
    from . import config

    os.makedirs(config.RUNS_DIR, exist_ok=True)
    return config.RUNS_DIR


def sanitize_basename(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")
    return cleaned or "resume"


def generate_slug(original_name: str) -> str:
    base = sanitize_basename(original_name)
    runs_dir = _runs_dir()
    candidate = base
    if os.path.isdir(os.path.join(runs_dir, candidate)):
        candidate = f"{base}_{uuid.uuid4().hex[:6]}"
    return candidate


def run_dir(slug: str) -> str:
    return os.path.join(_runs_dir(), slug)


def create_run(slug: str, *, source_pdf: str, original_filename: str, precision: str = "fp32") -> dict[str, Any]:
    out = run_dir(slug)
    os.makedirs(out, exist_ok=True)
    manifest: dict[str, Any] = {
        "slug": slug,
        "originalFilename": original_filename,
        "sourcePdf": source_pdf,
        "modelPrecision": precision,
        "status": "running",
        "currentStage": "extract_tokens",
        "failedStage": None,
        "error": None,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "completedAt": None,
        "artifacts": [],
    }
    write_manifest(slug, manifest)
    return manifest


def write_manifest(slug: str, manifest: dict[str, Any]) -> None:
    manifest["updatedAt"] = datetime.now(timezone.utc).isoformat()
    path = os.path.join(run_dir(slug), "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(slug: str) -> dict[str, Any] | None:
    path = os.path.join(run_dir(slug), "manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def update_stage(slug: str, stage: str) -> None:
    manifest = read_manifest(slug)
    if not manifest:
        return
    manifest["currentStage"] = stage
    write_manifest(slug, manifest)


def mark_completed(slug: str, artifacts: list[str], *, performance_stats: dict[str, Any] | None = None) -> None:
    manifest = read_manifest(slug) or {}
    manifest["status"] = "completed"
    manifest["currentStage"] = "done"
    manifest["artifacts"] = artifacts
    manifest["completedAt"] = datetime.now(timezone.utc).isoformat()
    if performance_stats is not None:
        manifest["performanceStats"] = performance_stats
    write_manifest(slug, manifest)


def mark_failed(slug: str, stage: str, error: str) -> None:
    manifest = read_manifest(slug) or {}
    manifest["status"] = "failed"
    manifest["failedStage"] = stage
    manifest["error"] = error
    write_manifest(slug, manifest)
    err_path = os.path.join(run_dir(slug), "error.txt")
    with open(err_path, "w", encoding="utf-8") as f:
        f.write(error)


def write_json(slug: str, filename: str, payload: Any) -> str:
    path = os.path.join(run_dir(slug), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    manifest = read_manifest(slug) or {}
    arts = manifest.setdefault("artifacts", [])
    if filename not in arts:
        arts.append(filename)
        write_manifest(slug, manifest)
    return path


def read_json(slug: str, filename: str) -> Any:
    path = os.path.join(run_dir(slug), filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def copy_input_pdf(slug: str, pdf_path: str) -> str:
    dest = os.path.join(run_dir(slug), "input.pdf")
    shutil.copy2(pdf_path, dest)
    return dest


def list_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    root = _runs_dir()
    if not os.path.isdir(root):
        return runs
    for name in os.listdir(root):
        manifest = read_manifest(name)
        if manifest:
            runs.append(manifest)
    runs.sort(key=lambda r: r.get("createdAt", ""), reverse=True)
    return runs


def artifact_path(slug: str, filename: str) -> str:
    safe = os.path.basename(filename)
    return os.path.join(run_dir(slug), safe)
