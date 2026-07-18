"""Shared sys.path helpers for training_pipeline lazy imports."""

from __future__ import annotations

import sys


def prioritize_training_paths(*dirs: str) -> None:
    """Move training dirs to the front of sys.path (first arg has highest priority)."""
    for directory in reversed(dirs):
        if directory in sys.path:
            sys.path.remove(directory)
        sys.path.insert(0, directory)
