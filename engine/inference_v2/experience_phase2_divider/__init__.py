"""Experience training phase 2 — section divider (job entry boundaries)."""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
from transformers import AutoTokenizer

from .config import MODEL_NAME, NUM_LABELS
from .model import build_segmenter
from .entry_postprocess import apply_entry_boundary_postprocess, demote_boundary_on_date_tokens
from .entry_style_heuristic import apply_style_entry_heuristic
from .heads_loader import load_entry_head_lines
from .entry_span_expand import expand_entry_span_labels
from .date_patterns import find_date_tokens

_ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from inference_v2.confidence import max_prob
from inference_v2.overlay_mongo_labels import overlay_mongo_field_labels  # noqa: E402
from inference_v2.model_precision import apply_precision  # noqa: E402
from inference_v2.predictor_cache import get_predictor  # noqa: E402

from data.base_dataset import _normalize_spatial  # noqa: E402
from vocab import EXP_BOUNDARY_LABELS, apply_sequence_constraints  # noqa: E402


class PyTorchExperiencePhase1Predictor:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )

        best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"Experience Phase 1 model not found: {best_model_path}")

        state_dict = torch.load(best_model_path, map_location=self.device, weights_only=True)
        spatial_dim = state_dict.get("spatial_fusion.spatial_mlp.0.weight").shape[1]
        has_crf = any(k.startswith("crf.") for k in state_dict.keys())

        self.model = build_segmenter(
            num_labels=NUM_LABELS,
            spatial_dim=spatial_dim,
            model_name=MODEL_NAME,
            use_crf=has_crf,
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()
        self.model, self.device = apply_precision(self.model, best_model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True, local_files_only=True)

    @torch.no_grad()
    def predict_tokens(self, filtered_tokens: list[dict]) -> tuple[list[str], list[float]]:
        if not filtered_tokens:
            return [], []

        words = [t["token"] for t in filtered_tokens]
        spatial = _normalize_spatial(filtered_tokens)

        total = len(words)
        word_preds = ["O"] * total
        word_confs = [0.0] * total
        word_best_dist = [float("inf")] * total

        window_size, overlap = 250, 100
        start = 0
        s_dim = self.model.spatial_dim

        while start < total:
            end = min(start + window_size, total)
            window_words = words[start:end]
            window_spatial = spatial[start:end]

            enc = self.tokenizer(
                window_words,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            seq_len = input_ids.shape[1]
            spatial_t = torch.zeros(1, seq_len, s_dim, dtype=torch.float32, device=self.device)
            word_ids = enc.word_ids(0)

            for i, wid in enumerate(word_ids):
                if wid is not None and wid < len(window_spatial):
                    spatial_t[0, i] = torch.tensor(
                        window_spatial[wid][:s_dim], dtype=torch.float32, device=self.device
                    )

            x0 = torch.tensor(
                [
                    window_spatial[wid][0] if (wid is not None and wid < len(window_spatial)) else 0.0
                    for wid in word_ids
                ],
                dtype=torch.float32,
                device=self.device,
            )
            y0 = torch.tensor(
                [
                    window_spatial[wid][1] if (wid is not None and wid < len(window_spatial)) else 0.0
                    for wid in word_ids
                ],
                dtype=torch.float32,
                device=self.device,
            )
            spatial_matrix = torch.stack([x0.unsqueeze(1) - x0.unsqueeze(0), y0.unsqueeze(1) - y0.unsqueeze(0)], dim=-1).unsqueeze(0)

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                spatial_features=spatial_t,
                spatial_matrix=spatial_matrix,
            )
            logits = apply_sequence_constraints(out["logits"], EXP_BOUNDARY_LABELS)
            preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
            logits_flat = logits.squeeze(0)

            window_mid = (start + end) / 2
            prev_wid = None
            for i, wid in enumerate(word_ids):
                if wid is None or wid >= len(window_words):
                    continue
                global_wid = start + wid
                if wid != prev_wid:
                    pred_idx = preds[i]
                    pred_label = EXP_BOUNDARY_LABELS[pred_idx] if pred_idx < len(EXP_BOUNDARY_LABELS) else "O"
                    dist = abs(global_wid - window_mid)
                    if dist < word_best_dist[global_wid] or pred_label != "O":
                        word_preds[global_wid] = pred_label
                        word_confs[global_wid] = max_prob(logits_flat[i], pred_idx)
                        word_best_dist[global_wid] = dist - 100 if pred_label != "O" else dist
                prev_wid = wid

            if end == total:
                break
            start += window_size - overlap

        return word_preds, word_confs


def run_experience_phase2_divider(tokens: list[dict], resume_id: str = "resume") -> dict[str, Any]:
    filtered_indices = [
        i for i, t in enumerate(tokens)
        if t.get("section") == "EXPERIENCE"
        and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]

    if len(filtered_indices) < 3:
        for idx in filtered_indices:
            tokens[idx]["bioLabel"] = "O"
            tokens[idx]["bio_label"] = "O"
        return _empty_output(tokens, filtered_indices, resume_id)

    filtered_tokens = [tokens[idx] for idx in filtered_indices]
    overlay_mongo_field_labels(filtered_tokens, resume_id)

    predictor = get_predictor("experience_phase2_divider", PyTorchExperiencePhase1Predictor)
    word_preds, word_confs = predictor.predict_tokens(filtered_tokens)
    for t, conf in zip(filtered_tokens, word_confs):
        t["confidence"] = conf

    head_lines = load_entry_head_lines(resume_id, tokens)
    if head_lines:
        exp_line_keys = {(t.get("page"), t.get("lineIndex", t.get("line_index"))) for t in filtered_tokens}
        head_lines = head_lines & exp_line_keys
        for t in filtered_tokens:
            key = (t.get("page"), t.get("lineIndex", t.get("line_index")))
            t["tempBoundaryLabel"] = "B-ENTRY" if key in head_lines else "O"
    else:
        raw_b_lines = sorted({
            (t.get("page"), t.get("lineIndex", t.get("line_index")))
            for t, pred in zip(filtered_tokens, word_preds)
            if pred == "B-ENTRY"
        })
        for t in filtered_tokens:
            key = (t.get("page"), t.get("lineIndex", t.get("line_index")))
            t["tempBoundaryLabel"] = "B-ENTRY" if key in raw_b_lines else "O"

    word_preds = apply_entry_boundary_postprocess(filtered_tokens, word_preds, use_bio_hints=True)
    word_preds = apply_style_entry_heuristic(filtered_tokens, word_preds)
    word_preds = expand_entry_span_labels(filtered_tokens, word_preds)
    word_preds = demote_boundary_on_date_tokens(filtered_tokens, word_preds)

    for t in filtered_tokens:
        t.pop("tempBoundaryLabel", None)

    for idx, pred in zip(filtered_indices, word_preds):
        tokens[idx]["bioLabel"] = pred
        tokens[idx]["bio_label"] = pred

    # Stamp durable entry-head markers so the structured grouping (build_entities_dict)
    # can reproduce these job boundaries — bioLabel gets overwritten by phase-1/phase-3.
    # Use the same filtered resolver as the UI (resolve_entry_slice_heads) so raw
    # per-token B-ENTRY on bullet/continuation lines does not create false entries.
    from inference_v2.experience_phase1_segment.entry_slice_heads import resolve_entry_slice_heads

    entry_head_keys = resolve_entry_slice_heads(filtered_tokens)
    for t in filtered_tokens:
        key = (int(t.get("page", 0)), int(t.get("lineIndex", t.get("line_index", 0))))
        if key in entry_head_keys:
            t["_expEntryHead"] = True

    non_o_count = len([p for p in word_preds if p != "O"])
    date_token_indices = find_date_tokens(filtered_tokens)

    return {
        "stage": "experience_phase2_divider",
        "title": "Experience Entry Boundaries",
        "section": "EXPERIENCE",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "experience/phase2_section_divider",
        "task": "entry_section_divider",
        "labelField": "bioLabel",
        "tokenCount": len(filtered_indices),
        "evalTokenCount": len(filtered_indices),
        "nonOCount": non_o_count,
        "dateTokenCount": len(date_token_indices),
        "sampleLabels": sorted(list(set(word_preds))),
        "tokens": [
            {
                "page": t["page"],
                "lineIndex": t["lineIndex"],
                "tokenIndex": t["tokenIndex"],
                "token": t["token"],
                "prediction": t["bioLabel"],
                "confidence": t.get("confidence", 0.0),
                "isDateToken": idx in date_token_indices,
                "x0": t.get("x0"),
                "y0": t.get("y0"),
                "x1": t.get("x1"),
                "y1": t.get("y1"),
            }
            for idx, t in enumerate(filtered_tokens)
        ],
    }


def _empty_output(tokens: list[dict], filtered_indices: list[int], resume_id: str) -> dict[str, Any]:
    return {
        "stage": "experience_phase2_divider",
        "title": "Experience Entry Boundaries",
        "section": "EXPERIENCE",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "experience/phase2_section_divider",
        "task": "entry_section_divider",
        "labelField": "bioLabel",
        "tokenCount": len(filtered_indices),
        "evalTokenCount": len(filtered_indices),
        "nonOCount": 0,
        "sampleLabels": [],
        "tokens": [
            {
                "page": tokens[idx]["page"],
                "lineIndex": tokens[idx]["lineIndex"],
                "tokenIndex": tokens[idx]["tokenIndex"],
                "token": tokens[idx]["token"],
                "prediction": "O",
                "x0": tokens[idx].get("x0"),
                "y0": tokens[idx].get("y0"),
                "x1": tokens[idx].get("x1"),
                "y1": tokens[idx].get("y1"),
            }
            for idx in filtered_indices
        ],
    }
