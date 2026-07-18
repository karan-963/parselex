#!/usr/bin/env python3
"""Client example: call the Parselex /inference-v2/parse API and print structured JSON.

With no args, parses the bundled demo resume (full-database/Karan.pdf) and saves
the result to examples/output/Karan.json.

Usage:
    python3 parse_resume.py
    python3 parse_resume.py resume.pdf
    python3 parse_resume.py resume.pdf --out result.json
    python3 parse_resume.py resume.pdf --url http://localhost:8000 --precision int8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PDF = os.path.join(_SCRIPT_DIR, "..", "full-database", "Karan.pdf")


def parse_resume(pdf_path: str, base_url: str, precision: str) -> dict:
    boundary = uuid.uuid4().hex
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{pdf_path.split("/")[-1]}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + pdf_bytes + f"\r\n--{boundary}--\r\n".encode()

    url = f"{base_url.rstrip('/')}/inference-v2/parse?precision={precision}"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", default=_DEFAULT_PDF, help="Path to resume PDF (default: bundled demo resume)")
    parser.add_argument("--url", default="http://localhost:8000", help="Parselex engine base URL")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
    parser.add_argument("--out", help="Write JSON output here (default: <script dir>/output/<pdf name>.json)")
    args = parser.parse_args()

    if not args.out:
        stem = os.path.splitext(os.path.basename(args.pdf))[0]
        args.out = os.path.join(_SCRIPT_DIR, "output", f"{stem}.json")

    try:
        result = parse_resume(args.pdf, args.url, args.precision)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode()}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Could not reach {args.url} — is the engine running? ({exc.reason})", file=sys.stderr)
        sys.exit(1)

    output = json.dumps(result["structured"], indent=2, ensure_ascii=False)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(output)
        print(f"Wrote structured JSON to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
