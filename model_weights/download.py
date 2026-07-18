#!/usr/bin/env python3
"""Cross-platform (Windows/macOS/Linux) download + extract for Parselex model weights.

Usage:
    python3 download.py
    PARSELEX_WEIGHTS_URL=https://... python3 download.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

URL = os.environ.get("PARSELEX_WEIGHTS_URL", "https://pub-6d31e47e899b48e69c7e7a45f99b3565.r2.dev/parselex-model-weights.zip")
MD5_EXPECTED = "1669fbcbf3bf772648f3fa4b5efd68d2"
ZIP_NAME = "parselex-model-weights.zip"


def main() -> None:
    if URL.startswith("<TODO"):
        print("error: URL not set. Edit download.py (or set PARSELEX_WEIGHTS_URL) before running.", file=sys.stderr)
        sys.exit(1)

    zip_path = os.path.join(SCRIPT_DIR, ZIP_NAME)
    print(f"Downloading {ZIP_NAME} from {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)

    print("Verifying MD5 ...")
    md5 = hashlib.md5()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    actual = md5.hexdigest()
    if actual != MD5_EXPECTED:
        print(f"error: MD5 mismatch!\n  expected: {MD5_EXPECTED}\n  actual:   {actual}", file=sys.stderr)
        print(f"Download is corrupt — delete {zip_path} and retry.", file=sys.stderr)
        sys.exit(1)
    print("MD5 OK.")

    print(f"Extracting into {SCRIPT_DIR} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(SCRIPT_DIR)

    print("Cleaning up zip ...")
    os.remove(zip_path)

    print("Done. Weights installed under model_weights/<stage>/.")


if __name__ == "__main__":
    main()
