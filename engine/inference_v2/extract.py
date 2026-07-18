"""PDF token extraction wrapper."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from . import config


def extract_tokens_from_pdf(pdf_path: str) -> tuple[list[dict], str]:
    spec = importlib.util.spec_from_file_location("extract_tokens", config.EXTRACT_TOKENS)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"extract_tokens.py not found: {config.EXTRACT_TOKENS}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.extract_tokens(pdf_path)
    tokens = result.get("tokens", [])
    resume_id = result.get("resumeId", Path(pdf_path).stem)
    return tokens, resume_id
