"""Hybrid heading detection: rule heuristics first, MiniLM on uncertain lines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer

from inference_v2.model_precision import apply_precision
from . import config
from .device_utils import resolve_device
from .heading_heuristics import (
    MAX_ALPHA_LEN,
    _alpha_len,
    _section_anchor_distance,
    compute_fha,
    is_false_heading,
    plain_line,
    predict_line_heading,
    predict_heading_lines as heuristic_heading_keys,
)
from .heading_postprocess import dedupe_adjacent_heading_keys
from .heading_style_match import propagate_style_headings
from .line_builder import LineRecord, build_parser_lines, gt_heading_keys
from .line_dataset import encode_line_text
from .line_features import build_line_samples
from .model_line_minilm import LineHeadingMiniLM


HEURISTIC_SKIP_CONF = config.HEURISTIC_SKIP_CONF
MODEL_HEADING_PROB = config.MODEL_HEADING_PROB


@dataclass
class LinePrediction:
    key: tuple[int, int]
    text: str
    is_heading: bool
    source: str  # heuristic | heuristic_reject | minilm | style_match | none
    heuristic_is_heading: bool
    heuristic_conf: float
    model_prob: float | None
    font_size: float
    is_bold: bool


class LineHybridPredictor:
    """Heuristics for high-confidence lines; MiniLM for the rest."""

    def __init__(self, model_path: str | None = None, device: torch.device | None = None):
        self.device = device or resolve_device()
        model_path = model_path or os.path.join(config.SAVED_MODELS_DIR, config.LINE_BEST_MODEL)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"MiniLM checkpoint not found: {model_path}")

        self.model = LineHeadingMiniLM()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device).eval()
        self.model, self.device = apply_precision(self.model, model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.LINE_MINILM_NAME, add_prefix_space=True, local_files_only=True)

    @torch.no_grad()
    def _model_prob(self, sample: dict[str, Any]) -> float:
        enc = encode_line_text(
            self.tokenizer,
            sample["prev_text"],
            sample["text"],
            sample["next_text"],
            config.LINE_MAX_SEQ_LEN,
        )
        spatial = torch.tensor([sample["spatial"]], dtype=torch.float32, device=self.device)
        logits = self.model(
            input_ids=enc["input_ids"].unsqueeze(0).to(self.device),
            attention_mask=enc["attention_mask"].unsqueeze(0).to(self.device),
            spatial=spatial,
        )["logits"]
        return torch.softmax(logits, dim=-1)[0, 1].item()

    def predict_document(self, tokens: list[dict[str, Any]]) -> list[LinePrediction]:
        lines = build_parser_lines(tokens)
        if not lines:
            return []

        fonts = [float(t.get("fontSize", 11.0) or 11.0) for t in tokens]
        doc_median_font = sorted(fonts)[len(fonts) // 2] if fonts else 11.0
        samples = {s["key"]: s for s in build_line_samples(tokens, lines)}
        skills_keys = frozenset({"skills", "technical skills", "core competencies", "areas of expertise"})
        work_keys = frozenset({
            "work history", "work experience", "professional experience", "experience", "employment",
        })
        results: list[LinePrediction] = []

        for i, line in enumerate(lines):
            raw = line.text
            plain = plain_line(raw)
            h_is, h_conf = predict_line_heading(
                line,
                doc_median_font,
                lines_since_skills=_section_anchor_distance(lines, i, skills_keys),
                lines_since_work_history=_section_anchor_distance(lines, i, work_keys),
            )
            m_prob: float | None = None
            source = "none"
            is_heading = False

            if _alpha_len(raw) > MAX_ALPHA_LEN or is_false_heading(plain, raw):
                source = "heuristic_reject"
            elif h_conf >= HEURISTIC_SKIP_CONF:
                is_heading = h_is
                source = "heuristic"
            else:
                sample = samples[line.key]
                m_prob = self._model_prob(sample)
                is_heading = m_prob >= MODEL_HEADING_PROB
                source = "minilm"

            results.append(
                LinePrediction(
                    key=line.key,
                    text=line.text,
                    is_heading=is_heading,
                    source=source,
                    heuristic_is_heading=h_is,
                    heuristic_conf=h_conf,
                    model_prob=m_prob,
                    font_size=line.font_median,
                    is_bold=line.is_bold,
                )
            )
        results = self._apply_style_propagation(lines, results)
        return self._apply_adjacent_dedupe(lines, results)

    def predict_model_document(self, tokens: list[dict[str, Any]]) -> list[LinePrediction]:
        """MiniLM on every non-rejected line (no heuristic skip)."""
        lines = build_parser_lines(tokens)
        if not lines:
            return []

        fonts = [float(t.get("fontSize", 11.0) or 11.0) for t in tokens]
        doc_median_font = sorted(fonts)[len(fonts) // 2] if fonts else 11.0
        samples = {s["key"]: s for s in build_line_samples(tokens, lines)}
        skills_keys = frozenset({"skills", "technical skills", "core competencies", "areas of expertise"})
        work_keys = frozenset({
            "work history", "work experience", "professional experience", "experience", "employment",
        })
        results: list[LinePrediction] = []

        for i, line in enumerate(lines):
            raw = line.text
            plain = plain_line(raw)
            h_is, h_conf = predict_line_heading(
                line,
                doc_median_font,
                lines_since_skills=_section_anchor_distance(lines, i, skills_keys),
                lines_since_work_history=_section_anchor_distance(lines, i, work_keys),
            )
            m_prob: float | None = None
            source = "none"
            is_heading = False

            if _alpha_len(raw) > MAX_ALPHA_LEN or is_false_heading(plain, raw):
                source = "heuristic_reject"
            else:
                sample = samples[line.key]
                m_prob = self._model_prob(sample)
                is_heading = m_prob >= MODEL_HEADING_PROB
                source = "minilm"

            results.append(
                LinePrediction(
                    key=line.key,
                    text=line.text,
                    is_heading=is_heading,
                    source=source,
                    heuristic_is_heading=h_is,
                    heuristic_conf=h_conf,
                    model_prob=m_prob,
                    font_size=line.font_median,
                    is_bold=line.is_bold,
                )
            )
        return self._apply_adjacent_dedupe(lines, results)

    def _apply_style_propagation(
        self, lines: list[LineRecord], results: list[LinePrediction]
    ) -> list[LinePrediction]:
        anchor_keys = {p.key for p in results if p.is_heading}
        candidate_keys = {p.key for p in results if not p.is_heading}
        promoted = propagate_style_headings(lines, anchor_keys, candidate_keys)
        if not promoted:
            return results

        out: list[LinePrediction] = []
        for p in results:
            if p.key in promoted:
                out.append(
                    LinePrediction(
                        key=p.key,
                        text=p.text,
                        is_heading=True,
                        source="style_match",
                        heuristic_is_heading=p.heuristic_is_heading,
                        heuristic_conf=p.heuristic_conf,
                        model_prob=p.model_prob,
                        font_size=p.font_size,
                        is_bold=p.is_bold,
                    )
                )
            else:
                out.append(p)
        return out

    def _apply_adjacent_dedupe(
        self, lines: list[LineRecord], results: list[LinePrediction]
    ) -> list[LinePrediction]:
        raw_keys = {p.key for p in results if p.is_heading}
        kept = dedupe_adjacent_heading_keys(lines, raw_keys)
        if kept == raw_keys:
            return results
        out: list[LinePrediction] = []
        for p in results:
            if p.is_heading and p.key not in kept:
                out.append(
                    LinePrediction(
                        key=p.key,
                        text=p.text,
                        is_heading=False,
                        source=p.source,
                        heuristic_is_heading=p.heuristic_is_heading,
                        heuristic_conf=p.heuristic_conf,
                        model_prob=p.model_prob,
                        font_size=p.font_size,
                        is_bold=p.is_bold,
                    )
                )
            else:
                out.append(p)
        return out

    def predict_heuristic_keys(self, tokens: list[dict[str, Any]]) -> set[tuple[int, int]]:
        return heuristic_heading_keys(tokens)

    def predict_model_keys(self, tokens: list[dict[str, Any]]) -> set[tuple[int, int]]:
        return {p.key for p in self.predict_model_document(tokens) if p.is_heading}

    def predict_line_keys(self, tokens: list[dict[str, Any]]) -> set[tuple[int, int]]:
        return {p.key for p in self.predict_document(tokens) if p.is_heading}


def gt_line_keys(tokens: list[dict[str, Any]]) -> set[tuple[int, int]]:
    return gt_heading_keys(tokens)
