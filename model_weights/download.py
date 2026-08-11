#!/usr/bin/env python3
"""Cross-platform (Windows/macOS/Linux) download for Parselex model weights, via Hugging Face.

Usage:
    python3 download.py
    PARSELEX_WEIGHTS_REPO=someone/fork-of-weights python3 download.py

Requires: pip install huggingface_hub
"""
from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_ID = os.environ.get("PARSELEX_WEIGHTS_REPO", "karan963/parselex-weights")


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("error: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading weights from https://huggingface.co/{REPO_ID} into {SCRIPT_DIR} ...")
    snapshot_download(repo_id=REPO_ID, local_dir=SCRIPT_DIR, allow_patterns=["*/*.pt"])

    print("Done. Weights installed under model_weights/<stage>/.")


if __name__ == "__main__":
    main()
