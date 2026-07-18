"""Project training phase 2 — entry boundary divider (B-PROJ_START / I-PROJ_START)."""

from __future__ import annotations

import os
import sys
from typing import Any
import torch
from transformers import AutoTokenizer

from .config import MODEL_NAME, NUM_LABELS, SPATIAL_DIM
from .model import build_segmenter
from .boundary_postprocess import apply_full_boundary_pipeline, is_project_segment
from .style_heuristic import apply_project_style_heuristic
from .heads_loader import load_entry_head_lines
from .span_expand import expand_project_span_labels
from .segment_utils import (
    clean_cid_tokens,
    construct_sentences_by_appearance,
    extract_16d_spatial,
    group_segments_by_line,
)
from .line_utils import build_physical_line_text_map
from .entry_divider_lines import build_entry_divider_line_rows

_ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from inference_v2.confidence import batch_max_probs
from inference_v2.overlay_mongo_labels import overlay_mongo_field_labels
from inference_v2.model_precision import apply_precision
from inference_v2.predictor_cache import get_predictor


class PyTorchProjectPhase2DividerPredictor:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )

        best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"Project phase2 divider model not found: {best_model_path}")

        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model = build_segmenter(
            num_labels=NUM_LABELS,
            spatial_dim=SPATIAL_DIM,
            model_name=MODEL_NAME,
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()
        self.model, self.device = apply_precision(self.model, best_model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True, local_files_only=True)

    @torch.no_grad()
    def predict_segments(self, segments: list[dict], spatial_features: list[list[float]]) -> tuple[list[int], list[float]]:
        if not segments:
            return [], []

        max_segs = 256
        max_seg_len = 32

        seg_input_ids = []
        seg_attn_mask = []

        seg_texts = [s["text"] for s in segments]
        for text in seg_texts[:max_segs]:
            enc = self.tokenizer(text, max_length=max_seg_len, padding="max_length", truncation=True, return_tensors="pt")
            seg_input_ids.append(enc["input_ids"].squeeze(0))
            seg_attn_mask.append(enc["attention_mask"].squeeze(0))

        num_segs = len(seg_texts)
        if num_segs < max_segs:
            for _ in range(max_segs - num_segs):
                seg_input_ids.append(torch.full((max_seg_len,), self.tokenizer.pad_token_id, dtype=torch.long))
                seg_attn_mask.append(torch.zeros(max_seg_len, dtype=torch.long))
                spatial_features.append([0.0] * 16)
        else:
            seg_input_ids = seg_input_ids[:max_segs]
            seg_attn_mask = seg_attn_mask[:max_segs]
            spatial_features = spatial_features[:max_segs]

        input_ids = torch.stack(seg_input_ids).unsqueeze(0).to(self.device)
        attention_mask = torch.stack(seg_attn_mask).unsqueeze(0).to(self.device)
        spatial_features_t = torch.tensor(spatial_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        logits = self.model(input_ids, attention_mask, spatial_features_t)
        active = min(num_segs, max_segs)
        preds = logits.squeeze(0).argmax(-1).cpu().tolist()[:active]
        confs = batch_max_probs(logits.squeeze(0)[:active], [int(p) for p in preds])
        return preds, confs


def run_project_phase2_divider(tokens: list[dict], resume_id: str = "resume") -> dict[str, Any]:
    # 1. Filter PROJECT/PROJECTS tokens (exclude B-HEADING/I-HEADING)
    filtered_indices = [
        i for i, t in enumerate(tokens)
        if t.get("section") in ("PROJECT", "PROJECTS")
        and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]

    if len(filtered_indices) < 3:
        for idx in filtered_indices:
            tokens[idx]["bioLabel"] = "O"
            tokens[idx]["bio_label"] = "O"
        return _empty_output(tokens, filtered_indices, resume_id)

    filtered_tokens = [tokens[idx] for idx in filtered_indices]
    overlay_mongo_field_labels(filtered_tokens, resume_id)

    # Reconstruct segments from filtered tokens
    cleaned_tokens = clean_cid_tokens(filtered_tokens)
    segments = construct_sentences_by_appearance(cleaned_tokens)

    if not segments:
        for idx in filtered_indices:
            tokens[idx]["bioLabel"] = "O"
            tokens[idx]["bio_label"] = "O"
        return _empty_output(tokens, filtered_indices, resume_id)

    # Extract 16D spatial features
    project_font_sizes = []
    for s in segments:
        if is_project_segment(s):
            if "spatial" in s and len(s["spatial"]) > 0:
                project_font_sizes.append(s["spatial"][0])
    if not project_font_sizes:
        project_font_sizes = [
            s["spatial"][0]
            for s in segments
            if "spatial" in s and len(s["spatial"]) > 0
        ]
    if not project_font_sizes:
        project_font_sizes = [10.0]

    max_size = max(project_font_sizes)
    min_size = min(project_font_sizes)
    from collections import Counter
    c = Counter(project_font_sizes)
    default_size = c.most_common(1)[0][0]

    spatial_features = [
        extract_16d_spatial(segments[i], segments[i-1] if i > 0 else None, max_size, default_size, min_size)
        for i in range(len(segments))
    ]

    predictor = get_predictor("project_phase2_divider", PyTorchProjectPhase2DividerPredictor)
    preds, seg_confs = predictor.predict_segments(segments, spatial_features)

    seg_token_conf: dict[int, float] = {}
    for seg_idx, segment in enumerate(segments[: len(preds)]):
        conf = seg_confs[seg_idx] if seg_idx < len(seg_confs) else 0.0
        for tok in segment.get("tokens", []):
            if tok:
                seg_token_conf[id(tok)] = conf

    num_segs = len(preds)
    groups_eval = group_segments_by_line(segments[:num_segs])

    # Build project lines coords set
    project_lines = set()
    for s in segments:
        if is_project_segment(s):
            for tok in s.get("tokens", []):
                if tok and "page" in tok and "lineIndex" in tok:
                    project_lines.add((tok["page"], tok["lineIndex"]))

    physical_lines = build_physical_line_text_map(segments, cleaned_tokens)

    ID2LABEL = {0: "O", 1: "B-PROJ_START", 2: "I-PROJ_START"}

    # Run boundary post-processing pipeline
    seg_preds, _, model_pred_line_set = apply_full_boundary_pipeline(
        preds, segments, num_segs, spatial_features[:num_segs], groups_eval,
        project_lines, ID2LABEL, is_project_segment,
        line_text_by_coord=physical_lines,
    )

    head_lines = load_entry_head_lines(resume_id)
    if head_lines:
        proj_line_keys = {(t.get("page"), t.get("lineIndex")) for t in filtered_tokens}
        pred_line_set = head_lines & proj_line_keys
    else:
        pred_line_set = model_pred_line_set

    # Initial token predictions
    word_preds = ["O"] * len(filtered_tokens)
    for idx, t in enumerate(filtered_tokens):
        key = (t.get("page"), t.get("lineIndex"))
        if key in pred_line_set:
            word_preds[idx] = "B-PROJ_START"

    # Expand spans
    word_preds = expand_project_span_labels(filtered_tokens, word_preds)

    for idx, pred in zip(filtered_indices, word_preds):
        tokens[idx]["bioLabel"] = pred
        tokens[idx]["bio_label"] = pred
        tokens[idx]["confidence"] = seg_token_conf.get(id(tokens[idx]), 0.0)

    # Stamp durable entry-head markers so structured grouping (PROJECT_ENTRIES)
    # can reproduce step-12 boundaries — bioLabel is overwritten by phase-3.
    for t in filtered_tokens:
        key = (int(t.get("page", 0)), int(t.get("lineIndex", 0)))
        if key in pred_line_set:
            t["_projEntryHead"] = True

    non_o_count = len([p for p in word_preds if p != "O"])
    entry_divider = build_entry_divider_line_rows(tokens, resume_id)

    return {
        "stage": "project_phase2_divider",
        "title": "Project Entry Boundaries",
        "section": "PROJECT",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "project/phase2_section_divider",
        "task": "entry_section_divider",
        "labelField": "prediction",
        "labels": ["O", "B-PROJ_START", "I-PROJ_START"],
        "entryDividerLines": entry_divider,
        "gtEntryHeadSource": "mongodb.projectEntryHeads",
        "tokenCount": len(filtered_indices),
        "evalTokenCount": len(filtered_indices),
        "nonOCount": non_o_count,
        "sampleLabels": sorted(list(set(word_preds))),
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


def _empty_output(tokens: list[dict], filtered_indices: list[int], resume_id: str) -> dict[str, Any]:
    entry_divider = build_entry_divider_line_rows(tokens, resume_id)

    return {
        "stage": "project_phase2_divider",
        "title": "Project Entry Boundaries",
        "section": "PROJECT",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "project/phase2_section_divider",
        "task": "entry_section_divider",
        "labelField": "prediction",
        "labels": ["O", "B-PROJ_START", "I-PROJ_START"],
        "entryDividerLines": entry_divider,
        "gtEntryHeadSource": "mongodb.projectEntryHeads",
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
