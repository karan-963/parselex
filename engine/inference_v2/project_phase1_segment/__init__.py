"""Project training phase 1 — token segmentation (phrase B-SEG / I-SEG).

Groups PROJECT tokens into entries using step-12 B-PROJ_START boundaries,
runs PhraseSegmenterTransformer per entry, applies sequence continuity guard only
(training parity — no gap_heuristic), and writes seg predictions.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import torch
from transformers import AutoTokenizer

from inference_v2.overlay_mongo_labels import load_mongo_entry_heads, overlay_mongo_field_labels
from inference_v2.model_precision import apply_precision
from inference_v2.predictor_cache import get_predictor

from .config import MODEL_NAME, NUM_LABELS
from .data_utils import _normalize_spatial
from .entry_slice_heads import resolve_project_entry_heads
from .model import build_segmenter

logger = logging.getLogger(__name__)

LABEL_MAP = ["O", "B-SEG", "I-SEG"]
STRUCTURAL_TOKENS = frozenset({"|", "•", "-", "–", "—", "*", "▪", "◦", "■", "·", ",", "✓", "✔", '"'})


def _derive_seg_gt(bio: str | None) -> str:
    if not bio or bio == "O" or "HEADING" in bio:
        return "O"
    return "B-SEG" if bio.startswith("B-") else "I-SEG"


def _is_eval_token(token_str: str) -> bool:
    tok = (token_str or "").strip()
    if tok in STRUCTURAL_TOKENS:
        return False
    return bool(re.search(r"[a-zA-Z0-9]", tok))


def _build_token_segmentation_metrics(filtered_tokens: list[dict]) -> dict[str, Any]:
    correct = 0
    eval_total = 0
    for t in filtered_tokens:
        if not _is_eval_token(t.get("token", "")):
            continue
        bio = t.get("_fieldBioLabel") or t.get("bioLabel") or "O"
        gt = _derive_seg_gt(bio)
        pred = t.get("prediction", "O")
        eval_total += 1
        if gt == pred:
            correct += 1
    accuracy = (correct / eval_total * 100.0) if eval_total else 0.0
    return {
        "gtSource": "mongodb.bioLabel→B-SEG/I-SEG",
        "trainingReport": "project/phase1_token_segmentation/reports/minilm/per_resume/*.md",
        "metrics": {
            "tokenAccuracyPercent": round(accuracy, 2),
            "correct": correct,
            "evalTokens": eval_total,
        },
    }


def _is_project_section(token: dict) -> bool:
    return token.get("section") in ("PROJECT", "PROJECTS")


def _description_lock_flags(entry_toks: list[dict]) -> list[bool]:
    flags = [False] * len(entry_toks)
    first_desc = -1
    last_desc = -1
    for idx, t in enumerate(entry_toks):
        bio = t.get("_fieldBioLabel") or t.get("bioLabel") or "O"
        if bio in ("B-DESC", "I-DESC"):
            if first_desc == -1:
                first_desc = idx
            last_desc = idx
    if first_desc != -1:
        for idx in range(first_desc, last_desc + 1):
            flags[idx] = True
    return flags


def _apply_continuity_guard(entry_toks: list[dict], preds: list[int]) -> list[int]:
    out = list(preds)
    for idx in range(1, len(entry_toks)):
        t_prev = entry_toks[idx - 1]
        t_curr = entry_toks[idx]
        same_line = (
            t_curr.get("page") == t_prev.get("page")
            and t_curr.get("lineIndex") == t_prev.get("lineIndex")
        )
        is_text = bool(re.search(r"[a-zA-Z0-9]", t_curr.get("token", ""))) and bool(
            re.search(r"[a-zA-Z0-9]", t_prev.get("token", ""))
        )
        if same_line and is_text and out[idx] == 1 and out[idx - 1] in (1, 2):
            out[idx] = 2
    return out


class PyTorchProjectPhase1Predictor:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )

        best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"Project phase1 segmenter model not found: {best_model_path}")

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
    def segment_entry(self, entry_tokens: list[dict]) -> tuple[list[int], list[float]]:
        if len(entry_tokens) < 3:
            return [2] * len(entry_tokens), [0.0] * len(entry_tokens)

        entry_words = []
        for t in entry_tokens:
            token_str = t.get("token", "")
            cleaned = token_str.replace('"', " ").replace("'", " ").replace("‘", " ").replace("’", " ")
            entry_words.append(cleaned)

        entry_spatial = [t["_temp_spatial"] for t in entry_tokens]

        enc = self.tokenizer(entry_words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        s_dim = getattr(self.model, "spatial_dim", 12)
        spatial_t = torch.zeros(1, input_ids.shape[1], s_dim).to(self.device)
        word_ids = enc.word_ids(0)

        for i, wid in enumerate(word_ids):
            if wid is not None and wid < len(entry_spatial):
                spatial_t[0, i] = torch.tensor(entry_spatial[wid][:s_dim])

        x0 = torch.tensor(
            [entry_spatial[wid][0] if (wid is not None and wid < len(entry_spatial)) else 0.0 for wid in word_ids],
            dtype=torch.float32,
            device=self.device,
        )
        y0 = torch.tensor(
            [entry_spatial[wid][1] if (wid is not None and wid < len(entry_spatial)) else 0.0 for wid in word_ids],
            dtype=torch.float32,
            device=self.device,
        )
        dx = x0.unsqueeze(1) - x0.unsqueeze(0)
        dy = y0.unsqueeze(1) - y0.unsqueeze(0)
        spatial_matrix = torch.stack([dx, dy], dim=-1).unsqueeze(0)

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            spatial_features=spatial_t,
            spatial_matrix=spatial_matrix,
        )
        preds = logits[0].argmax(dim=-1)
        probs = torch.softmax(logits[0], dim=-1)

        entry_preds: dict[int, int] = {}
        entry_conf: dict[int, float] = {}
        for i, wid in enumerate(word_ids):
            if wid is not None and wid < len(entry_tokens) and wid not in entry_preds:
                entry_preds[wid] = int(preds[i])
                entry_conf[wid] = float(probs[i, preds[i]].item())

        labels = [entry_preds.get(wid, 2) for wid in range(len(entry_tokens))]
        confs = [entry_conf.get(wid, 0.0) for wid in range(len(entry_tokens))]
        return labels, confs


def _group_entries(filtered_tokens: list[dict], head_lines: set[tuple[int, int]]) -> list[tuple[tuple[int, int] | None, list[dict]]]:
    sorted_heads = sorted(head_lines)
    entries: list[tuple[tuple[int, int] | None, list[dict]]] = []
    current_block: list[dict] = []
    current_head: tuple[int, int] | None = None

    for t in filtered_tokens:
        key = (t["page"], t["lineIndex"])
        if key in head_lines and key != current_head:
            if current_block:
                entries.append((current_head, current_block))
                current_block = []
            current_head = key
        if current_head is not None or not sorted_heads:
            current_block.append(t)

    if current_block:
        entries.append((current_head, current_block))
    return entries


def run_project_phase1_segment(tokens: list[dict], resume_id: str = "resume") -> dict[str, Any]:
    filtered_indices = [
        i for i, t in enumerate(tokens)
        if _is_project_section(t) and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]
    if not filtered_indices:
        return _empty_result(resume_id)

    filtered_tokens = [tokens[idx] for idx in filtered_indices]
    overlay_mongo_field_labels(filtered_tokens, resume_id)

    spatial_all = _normalize_spatial(filtered_tokens)
    for t, spat in zip(filtered_tokens, spatial_all):
        s = list(spat)
        if len(s) < 12:
            s = s + [0.0] * (12 - len(s))
        s[11] = 0.0
        t["_temp_spatial"] = s

    head_lines = resolve_project_entry_heads(filtered_tokens)
    if not head_lines:
        mongo_heads = load_mongo_entry_heads(resume_id, "PROJECT")
        proj_keys = {(t.get("page"), t.get("lineIndex")) for t in filtered_tokens}
        head_lines = mongo_heads & proj_keys
        if head_lines:
            logger.warning(
                "project_phase1_segment: no B-PROJ_START lines — falling back to mongo projectEntryHeads",
            )
        else:
            logger.warning("project_phase1_segment: no entry heads found — treating section as one block")
            head_lines = set()

    entries = _group_entries(filtered_tokens, head_lines)
    predictor = get_predictor("project_phase1_segment", PyTorchProjectPhase1Predictor)

    for _head, entry_toks in entries:
        if len(entry_toks) < 3:
            for t in entry_toks:
                t["segLabel"] = "O"
            continue

        desc_flags = _description_lock_flags(entry_toks)
        for t, is_desc in zip(entry_toks, desc_flags):
            if "_temp_spatial" in t and len(t["_temp_spatial"]) >= 12:
                t["_temp_spatial"][11] = float(is_desc)

        entry_preds, entry_conf_list = predictor.segment_entry(entry_toks)
        for t, conf in zip(entry_toks, entry_conf_list):
            t["_segConfidence"] = conf
        entry_preds = _apply_continuity_guard(entry_toks, entry_preds)

        for t, pred_id in zip(entry_toks, entry_preds):
            t["segLabel"] = LABEL_MAP[pred_id]

    for t in filtered_tokens:
        t.pop("_temp_spatial", None)
        seg = t.get("segLabel", "O")
        t["prediction"] = seg
        t["segLabel"] = seg
        t["segConfidence"] = round(float(t.pop("_segConfidence", 0.0)), 4)

    non_o_count = sum(1 for t in filtered_tokens if t.get("prediction") != "O")
    token_segmentation = _build_token_segmentation_metrics(filtered_tokens)

    return {
        "stage": "project_phase1_segment",
        "title": "Project Token Segmentation",
        "section": "PROJECT",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "project/phase1_token_segmentation",
        "task": "token_segmentation",
        "phraseSegmentationPipeline": "project/phase1_token_segmentation",
        "labelField": "prediction",
        "labels": LABEL_MAP,
        "tokenSegmentation": token_segmentation,
        "gtSegSource": "mongodb.bioLabel→B-SEG/I-SEG",
        "tokenCount": len(filtered_tokens),
        "evalTokenCount": len(filtered_tokens),
        "nonOCount": non_o_count,
        "sampleLabels": sorted({t.get("prediction", "O") for t in filtered_tokens}),
        "tokens": [
            {
                "page": t["page"],
                "lineIndex": t["lineIndex"],
                "tokenIndex": t["tokenIndex"],
                "token": t["token"],
                "prediction": t["prediction"],
                "confidence": t.get("segConfidence", 0.0),
                "x0": t.get("x0"),
                "y0": t.get("y0"),
                "x1": t.get("x1"),
                "y1": t.get("y1"),
            }
            for t in filtered_tokens
        ],
    }


def _empty_result(resume_id: str) -> dict[str, Any]:
    return {
        "stage": "project_phase1_segment",
        "title": "Project Token Segmentation",
        "section": "PROJECT",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "project/phase1_token_segmentation",
        "task": "token_segmentation",
        "phraseSegmentationPipeline": "project/phase1_token_segmentation",
        "labelField": "prediction",
        "labels": LABEL_MAP,
        "tokenCount": 0,
        "evalTokenCount": 0,
        "nonOCount": 0,
        "sampleLabels": [],
        "tokens": [],
    }
