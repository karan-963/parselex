"""Section Phase 2 — PyTorch section classifier stage."""

from __future__ import annotations

import os
import torch
from typing import Any
from transformers import AutoTokenizer

from .config import MODEL_NAME, MAX_LENGTH, NUM_CLASSES, SPATIAL_DIM
from .model import ResumeSectionClassifier
from .dataset import (
    sort_tokens_by_reading_order,
    generate_spatial_features,
    median_inter_token_gap,
    median_line_height_of,
    _aggregate_chunks,
    ID2CHUNK,
)
from .strategy import post_process_predictions
from ..predictor_cache import get_predictor
from ..model_precision import apply_precision


def _pad_ids(enc, tokenizer, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    seq_len = input_ids.shape[1]
    if seq_len > max_length:
        input_ids = torch.cat([input_ids[:, :128], input_ids[:, -128:]], dim=1)
        attention_mask = torch.cat([attention_mask[:, :128], attention_mask[:, -128:]], dim=1)
    else:
        pad_len = max_length - seq_len
        input_ids = torch.cat(
            [input_ids, torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)], dim=1
        )
        attention_mask = torch.cat([attention_mask, torch.zeros((1, pad_len), dtype=torch.long)], dim=1)
    return input_ids, attention_mask


class PyTorchSectionPhase2Predictor:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        self.model = ResumeSectionClassifier(
            num_classes=NUM_CLASSES, spatial_dim=SPATIAL_DIM, model_name=MODEL_NAME
        )
        
        best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"PyTorch model not found: {best_model_path}")
        
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()
        self.model, self.device = apply_precision(self.model, best_model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    @torch.no_grad()
    def predict_chunk(self, text: str, spatial_feat: list[float], prev_label: int) -> tuple[int, float]:
        enc = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = _pad_ids(enc, self.tokenizer, MAX_LENGTH)
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        spatial_features = torch.tensor([spatial_feat], dtype=torch.float32, device=self.device)
        prev_labels = torch.tensor([prev_label], dtype=torch.long, device=self.device)
        
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            spatial_features=spatial_features,
            prev_labels=prev_labels
        )
        
        logits = out["logits"][0]
        probs = torch.softmax(logits, dim=-1)
        prob, pred_id = probs.max(-1)
        return pred_id.item(), prob.item()


def run_section_phase2(tokens: list[dict]) -> dict[str, Any]:
    predictor = get_predictor("section_p2", PyTorchSectionPhase2Predictor)

    tokens[:] = sort_tokens_by_reading_order(tokens)
    line_map: dict[tuple, list] = {}
    for t in tokens:
        key = (t.get("page", 0), t.get("lineIndex", 0))
        line_map.setdefault(key, []).append(t)
        
    spatial = generate_spatial_features(
        tokens,
        median_inter_token_gap(line_map),
        median_line_height_of(tokens),
        augment=False,
    )
    token_idx_map = {id(t): i for i, t in enumerate(tokens)}
    prev_label = NUM_CLASSES
    chunks_meta: list[dict] = []
    aggregated = _aggregate_chunks(tokens)

    for chunk in aggregated:
        chunk_tokens = chunk["tokens"]
        heading_parts = [
            t["token"] for t in chunk_tokens if t.get("bioLabel") in ("B-HEADING", "I-HEADING")
        ]
        heading_text = " ".join(heading_parts) if heading_parts else chunk["heading"]
        heading_idx = token_idx_map.get(id(chunk_tokens[0]), 0) if chunk_tokens else 0

        if chunk.get("virtual", False):
            text = "[SECTION_START] " + " ".join(t["token"] for t in chunk_tokens if t["token"].strip())
        else:
            text = " ".join(t["token"] for t in chunk_tokens if t["token"].strip())

        pred_id, prob = predictor.predict_chunk(text, spatial[heading_idx], prev_label)
        pred_label = ID2CHUNK[pred_id]

        chunks_meta.append({
            "chunk": chunk,
            "text": text,
            "pred_label": pred_label,
            "confidence": prob,
        })
        prev_label = pred_id

    # Apply strategy post-processing
    final_preds = post_process_predictions(
        [c["chunk"] for c in chunks_meta],
        [c["pred_label"] for c in chunks_meta],
        [c["confidence"] for c in chunks_meta],
    )

    # Propagate predictions to tokens (including original tokens from outer scope)
    token_lookup = {
        (t.get("page"), t.get("lineIndex"), t.get("tokenIndex")): t
        for t in tokens
    }
    for meta, final_lbl in zip(chunks_meta, final_preds):
        meta["final_label"] = final_lbl
        for t in meta["chunk"]["tokens"]:
            t["section"] = final_lbl
            orig_t = token_lookup.get((t.get("page"), t.get("lineIndex"), t.get("tokenIndex")))
            if orig_t:
                orig_t["section"] = final_lbl

    return {
        "stage": "section_phase2",
        "chunkCount": len(chunks_meta),
        "chunks": [
            {
                "heading": c["chunk"]["heading"],
                "text": c["text"],
                "prediction": c["pred_label"],
                "final_prediction": c["final_label"],
                "section": c["final_label"],
                "confidence": round(c["confidence"], 4),
                "tokenCount": len(c["chunk"]["tokens"]),
            }
            for c in chunks_meta
        ],
    }
