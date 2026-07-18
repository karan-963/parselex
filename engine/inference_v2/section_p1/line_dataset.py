"""PyTorch dataset: one sample = one parser line for MiniLM heading classification."""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from . import config
from .data_utils import sort_tokens_by_reading_order
from .excluded_resumes import is_excluded, list_active_json_files, load_excluded_ids
from .line_builder import build_parser_lines
from .line_features import SPATIAL_DIM, build_line_samples


def encode_line_text(
    tokenizer: AutoTokenizer,
    prev_text: str,
    text: str,
    next_text: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Encode prev / current / next line context as a single sequence."""
    parts: list[str] = []
    if prev_text.strip():
        parts.append(prev_text.strip())
    parts.append(text.strip() or ".")
    if next_text.strip():
        parts.append(next_text.strip())
    combined = " [SEP] ".join(parts)
    enc = tokenizer(
        combined,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in enc.items()}


def _load_line_samples(data_dir: str, split: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    excluded = load_excluded_ids()
    for fpath in list_active_json_files(data_dir, excluded):
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        doc_id = data.get("resumeId", os.path.basename(fpath))
        if is_excluded(doc_id, excluded):
            continue
        tokens = sort_tokens_by_reading_order(data.get("tokens", []))
        if len(tokens) < 5:
            continue
        lines = build_parser_lines(tokens)
        for s in build_line_samples(tokens, lines):
            s["doc_id"] = doc_id
            samples.append(s)
    pos = sum(1 for s in samples if s["label"] == 1)
    neg = len(samples) - pos
    print(f"[LINE-DATA] {split}: {len(samples)} lines ({pos} heading, {neg} other)")
    return samples


class MiniLMLineDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        self.samples = _load_line_samples(data_dir, split)
        self.tokenizer = AutoTokenizer.from_pretrained(config.LINE_MINILM_NAME, add_prefix_space=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        enc = encode_line_text(
            self.tokenizer,
            s["prev_text"],
            s["text"],
            s["next_text"],
            config.LINE_MAX_SEQ_LEN,
        )
        spatial = torch.tensor(s["spatial"], dtype=torch.float32)
        assert len(s["spatial"]) == SPATIAL_DIM
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "spatial": spatial,
            "labels": torch.tensor(s["label"], dtype=torch.long),
        }
