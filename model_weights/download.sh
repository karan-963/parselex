#!/usr/bin/env bash
# Download and extract Parselex model weights from Cloudflare storage.
# Run from anywhere — always installs into model_weights/ next to this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

URL="${PARSELEX_WEIGHTS_URL:-https://pub-6d31e47e899b48e69c7e7a45f99b3565.r2.dev/parselex-model-weights.zip}"
MD5_EXPECTED="1669fbcbf3bf772648f3fa4b5efd68d2"
ZIP_NAME="parselex-model-weights.zip"

if [[ "$URL" == "<TODO"* ]]; then
  echo "error: URL not set. Edit download.sh (or export PARSELEX_WEIGHTS_URL) before running." >&2
  exit 1
fi

echo "Downloading $ZIP_NAME from $URL ..."
curl -fL --progress-bar -o "$ZIP_NAME" "$URL"

echo "Verifying MD5 ..."
if command -v md5sum >/dev/null 2>&1; then
  MD5_ACTUAL="$(md5sum "$ZIP_NAME" | awk '{print $1}')"
else
  MD5_ACTUAL="$(md5 -q "$ZIP_NAME")"
fi

if [[ "$MD5_ACTUAL" != "$MD5_EXPECTED" ]]; then
  echo "error: MD5 mismatch!" >&2
  echo "  expected: $MD5_EXPECTED" >&2
  echo "  actual:   $MD5_ACTUAL" >&2
  echo "Download is corrupt — delete $ZIP_NAME and retry." >&2
  exit 1
fi
echo "MD5 OK."

echo "Extracting into $SCRIPT_DIR ..."
unzip -oq "$ZIP_NAME" -d "$SCRIPT_DIR"

echo "Cleaning up zip ..."
rm -f "$ZIP_NAME"

echo "Done. Weights installed under model_weights/<stage>/."
