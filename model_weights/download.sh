#!/usr/bin/env bash
# Download Parselex model weights from Hugging Face.
# Run from anywhere — always installs into model_weights/ next to this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ID="${PARSELEX_WEIGHTS_REPO:-karan963/parselex-weights}"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: 'hf' CLI not found. Install it with: pip install -U huggingface_hub" >&2
  exit 1
fi

echo "Downloading weights from https://huggingface.co/$REPO_ID into $SCRIPT_DIR ..."
hf download "$REPO_ID" --local-dir "$SCRIPT_DIR" --include "*/*.pt"

echo "Done. Weights installed under model_weights/<stage>/."
