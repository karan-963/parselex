"""Section Phase 1 — PyTorch hybrid heading detection stage."""

from __future__ import annotations

import os
import sys
from typing import Any

from .data_utils import sort_tokens_by_reading_order
from .heading_outlier_filter import find_outlier_heading_keys
from .line_builder import build_parser_lines
from .line_hybrid_predict import LineHybridPredictor
from ..predictor_cache import get_predictor


def apply_heading_predictions(tokens: list[dict], predictions) -> None:
    lines = build_parser_lines(tokens)
    line_by_key = {line.key: line for line in lines}

    for t in tokens:
        t["bioLabel"] = "O"
        t["bio_label"] = "O"

    for pred in predictions:
        if not pred.is_heading:
            continue
        line = line_by_key.get(pred.key)
        if line is None:
            continue
        for idx, ti in enumerate(line.token_indices):
            lbl = "B-HEADING" if idx == 0 else "I-HEADING"
            tokens[ti]["bioLabel"] = lbl
            tokens[ti]["bio_label"] = lbl


def run_section_phase1(tokens: list[dict]) -> dict[str, Any]:
    # Sort tokens in place
    tokens[:] = sort_tokens_by_reading_order(tokens)
    
    # Initialize the PyTorch predictor from the local directory
    # Note: LineHybridPredictor will load best_model_line_minilm.pt from the same folder
    predictor = get_predictor("section_p1", LineHybridPredictor)
    preds = predictor.predict_document(tokens)
    apply_heading_predictions(tokens, preds)

    # Build line lookup so we can resolve y0/x0 for each heading
    lines = build_parser_lines(tokens)
    line_by_key = {line.key: line for line in lines}

    # Drop style-outlier headings (e.g. bulleted "• Soft Skills :" sub-labels)
    # and demote their tokens so phase-2 merges them into the surrounding section.
    heading_records = [
        line_by_key[p.key] for p in preds if p.is_heading and p.key in line_by_key
    ]
    outlier_keys = find_outlier_heading_keys(heading_records)
    for line in lines:
        if line.key in outlier_keys:
            for ti in line.token_indices:
                tokens[ti]["bioLabel"] = "O"
                tokens[ti]["bio_label"] = "O"

    heading_lines = []
    for p in preds:
        if not p.is_heading or p.key in outlier_keys:
            continue
        line = line_by_key.get(p.key)
        y0 = None
        x0 = None
        if line and line.token_indices:
            first_tok = tokens[line.token_indices[0]]
            y0 = first_tok.get("y0")
            x0 = first_tok.get("x0")

        heading_lines.append({
            "page": p.key[0],
            "lineIndex": p.key[1],
            "text": p.text,
            "source": p.source,
            "heuristic_conf": round(p.heuristic_conf, 4),
            "model_prob": round(p.model_prob, 4) if p.model_prob is not None else None,
            "confidence": round(p.model_prob, 4) if p.model_prob is not None else round(p.heuristic_conf, 4),
            "y0": round(y0, 2) if y0 is not None else None,
            "x0": round(x0, 2) if x0 is not None else None,
        })

    heading_lines.sort(key=lambda h: (h["page"], h["lineIndex"]))
    return {
        "stage": "section_phase1",
        "headingCount": len(heading_lines),
        "headings": heading_lines,
    }
