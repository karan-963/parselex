"""Education training phase 2 — entry boundary divider (B-EDU_START / I-EDU_START)."""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoTokenizer

from inference_v2.confidence import batch_max_probs
from inference_v2.model_precision import apply_precision
from inference_v2.predictor_cache import get_predictor

from .config import ID2LABEL, MAX_EVAL_SEGS, MAX_SEG_LEN, MODEL_NAME, NUM_LABELS
from .entry_divider_lines import build_entry_divider_line_rows
from .model import build_segmenter
from .training_bridge import load_training_helpers
from .y0_line_collapse import collapse_lines_by_y0


class PyTorchEducationPhase2DividerPredictor:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )

        best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"Education phase2 divider model not found: {best_model_path}")

        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        self.model = build_segmenter(num_labels=NUM_LABELS, model_name=MODEL_NAME)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()
        self.model, self.device = apply_precision(self.model, best_model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True, local_files_only=True)

    @torch.no_grad()
    def predict_segments(self, segments: list[dict], spatial_features: list[list[float]]) -> tuple[list[int], list[float]]:
        if not segments:
            return [], []

        seg_input_ids: list[torch.Tensor] = []
        seg_attn_mask: list[torch.Tensor] = []
        seg_texts = [s["text"] for s in segments]

        for text in seg_texts[:MAX_EVAL_SEGS]:
            enc = self.tokenizer(text, max_length=MAX_SEG_LEN, padding="max_length", truncation=True, return_tensors="pt")
            seg_input_ids.append(enc["input_ids"].squeeze(0))
            seg_attn_mask.append(enc["attention_mask"].squeeze(0))

        num_segs = len(seg_texts)
        spatial = list(spatial_features)
        if num_segs < MAX_EVAL_SEGS:
            for _ in range(MAX_EVAL_SEGS - num_segs):
                seg_input_ids.append(torch.full((MAX_SEG_LEN,), self.tokenizer.pad_token_id, dtype=torch.long))
                seg_attn_mask.append(torch.zeros(MAX_SEG_LEN, dtype=torch.long))
        else:
            seg_input_ids = seg_input_ids[:MAX_EVAL_SEGS]
            seg_attn_mask = seg_attn_mask[:MAX_EVAL_SEGS]
            spatial = spatial[:MAX_EVAL_SEGS]

        from .training_bridge import load_training_helpers
        pad_spatial = load_training_helpers()["pad_spatial_features"]
        spatial = pad_spatial(spatial, MAX_EVAL_SEGS)

        input_ids = torch.stack(seg_input_ids).unsqueeze(0).to(self.device)
        attention_mask = torch.stack(seg_attn_mask).unsqueeze(0).to(self.device)
        spatial_features_t = torch.tensor(spatial, dtype=torch.float32).unsqueeze(0).to(self.device)

        logits = self.model(input_ids, attention_mask, spatial_features_t)
        active = min(num_segs, MAX_EVAL_SEGS)
        preds = logits.squeeze(0).argmax(-1).cpu().tolist()[:active]
        confs = batch_max_probs(logits.squeeze(0)[:active], [int(p) for p in preds])
        return preds, confs


def _education_filtered_indices(tokens: list[dict]) -> list[int]:
    return [
        i for i, t in enumerate(tokens)
        if t.get("section") == "EDUCATION"
        and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]


def _line_level_preds(filtered_tokens: list[dict], pred_line_set: set[tuple[int, int]]) -> list[str]:
    line_first: set[tuple[int, int]] = set()
    word_preds: list[str] = []
    for t in filtered_tokens:
        key = (int(t["page"]), int(t["lineIndex"]))
        if key in pred_line_set and key not in line_first:
            word_preds.append("B-EDU_START")
            line_first.add(key)
        else:
            word_preds.append("O")
    return word_preds


def _empty_output(tokens: list[dict], filtered_indices: list[int], resume_id: str) -> dict[str, Any]:
    entry_divider = build_entry_divider_line_rows(tokens, resume_id)
    return {
        "stage": "education_phase2_divider",
        "title": "Education Entry Boundaries",
        "section": "EDUCATION",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "education/new_phase2_section_divider",
        "task": "entry_section_divider",
        "labelField": "prediction",
        "labels": ["O", "B-EDU_START", "I-EDU_START"],
        "entryDividerLines": entry_divider,
        "gtEntryHeadSource": "mongodb.educationEntryHeads",
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


def run_education_phase2_divider(tokens: list[dict], resume_id: str = "resume") -> dict[str, Any]:
    helpers = load_training_helpers()
    filtered_indices = _education_filtered_indices(tokens)

    if not filtered_indices:
        return _empty_output(tokens, filtered_indices, resume_id)

    cleaned_all = helpers["clean_cid_tokens"](tokens)
    segments = helpers["construct_sentences_by_appearance"](cleaned_all)
    if not segments:
        return _empty_output(tokens, filtered_indices, resume_id)

    spatial_features = helpers["build_segment_spatial_features"](
        segments,
        is_education_segment=helpers["is_education_segment"],
        raw_tokens=cleaned_all,
    )

    predictor = get_predictor("education_phase2_divider", PyTorchEducationPhase2DividerPredictor)
    preds, seg_confs = predictor.predict_segments(segments, spatial_features)

    seg_token_conf: dict[int, float] = {}
    for seg_idx, segment in enumerate(segments[: len(preds)]):
        conf = seg_confs[seg_idx] if seg_idx < len(seg_confs) else 0.0
        for tok in segment.get("tokens", []):
            if tok:
                seg_token_conf[id(tok)] = conf

    num_segs = len(preds)
    groups_eval = helpers["group_segments_by_line"](segments[:num_segs])
    education_lines = helpers["collect_education_line_coords"](segments)
    physical_lines = helpers["build_physical_line_text_map"](segments, cleaned_all)

    _, _, pred_line_set = helpers["apply_full_boundary_pipeline"](
        preds,
        segments,
        num_segs,
        groups_eval,
        education_lines,
        ID2LABEL,
        helpers["is_education_segment"],
        line_text_by_coord=physical_lines,
    )

    filtered_tokens = [tokens[idx] for idx in filtered_indices]
    pred_line_set = collapse_lines_by_y0(filtered_tokens, pred_line_set)
    word_preds = _line_level_preds(filtered_tokens, pred_line_set)

    for idx, pred in zip(filtered_indices, word_preds):
        tokens[idx]["bioLabel"] = pred
        tokens[idx]["bio_label"] = pred
        if pred == "B-EDU_START":
            tokens[idx]["_eduEntryHead"] = True
        tokens[idx]["confidence"] = seg_token_conf.get(id(tokens[idx]), 0.0)

    non_o_count = sum(1 for p in word_preds if p != "O")
    entry_divider = build_entry_divider_line_rows(tokens, resume_id)

    return {
        "stage": "education_phase2_divider",
        "title": "Education Entry Boundaries",
        "section": "EDUCATION",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "education/new_phase2_section_divider",
        "task": "entry_section_divider",
        "labelField": "prediction",
        "labels": ["O", "B-EDU_START", "I-EDU_START"],
        "entryDividerLines": entry_divider,
        "gtEntryHeadSource": "mongodb.educationEntryHeads",
        "tokenCount": len(filtered_indices),
        "evalTokenCount": len(filtered_indices),
        "nonOCount": non_o_count,
        "sampleLabels": sorted(set(word_preds)),
        "tokens": [
            {
                "page": t["page"],
                "lineIndex": t["lineIndex"],
                "tokenIndex": t["tokenIndex"],
                "token": t["token"],
                "prediction": t["bioLabel"],
                "confidence": t.get("confidence", 0.0),
                "x0": t.get("x0"),
                "y0": t.get("y0"),
                "x1": t.get("x1"),
                "y1": t.get("y1"),
            }
            for t in filtered_tokens
        ],
    }
