"""Experience training phase 1 — token segmentation (phrase B-SEG / I-SEG).

Groups EXPERIENCE tokens into entries (using predicted B-ENTRY boundaries),
runs PhraseSegmenterTransformer on each entry to predict phrase boundaries (B-SEG/I-SEG/O),
applies programmatic continuity guards and gap/date heuristics,
and writes predictions back.
"""

from __future__ import annotations

import os
import re
import torch
from typing import Any
from transformers import AutoTokenizer

from .config import MODEL_NAME, SPATIAL_DIM, NUM_LABELS, MAX_LENGTH
from .model import build_segmenter
from .data_utils import _normalize_spatial, clean_non_text_tokens
from .gap_heuristic import apply_segment_postprocess
from .structural_heuristic import apply_structural_segmentation
from .entry_slice_heads import resolve_segmentation_entry_heads
from inference_v2.overlay_mongo_labels import overlay_mongo_field_labels
from inference_v2.model_precision import apply_precision
from inference_v2.predictor_cache import get_predictor


class PyTorchExperiencePhase2Predictor:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        
        best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"Experience Phase 2 segmenter model not found: {best_model_path}")
            
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
            cleaned_token = token_str.replace('"', '').replace("'", "").replace("‘", "").replace("’", "")
            entry_words.append(cleaned_token)
            
        entry_spatial = [t["_temp_spatial"] for t in entry_tokens]
        
        enc = self.tokenizer(entry_words, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True)
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)
        
        s_dim = getattr(self.model, "spatial_dim", 12)
        spatial_t = torch.zeros(1, input_ids.shape[1], s_dim).to(self.device)
        word_ids = enc.word_ids(0)
        
        for i, wid in enumerate(word_ids):
            if wid is not None and wid < len(entry_spatial):
                spatial_t[0, i] = torch.tensor(entry_spatial[wid][:s_dim])
                
        x0 = torch.tensor([entry_spatial[wid][0] if (wid is not None and wid < len(entry_spatial)) else 0.0 for wid in word_ids], dtype=torch.float32, device=self.device)
        y0 = torch.tensor([entry_spatial[wid][1] if (wid is not None and wid < len(entry_spatial)) else 0.0 for wid in word_ids], dtype=torch.float32, device=self.device)
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

        entry_preds = {}
        entry_conf: dict[int, float] = {}
        for i, wid in enumerate(word_ids):
            if wid is not None and wid < len(entry_tokens):
                if wid not in entry_preds:
                    entry_preds[wid] = preds[i]
                    entry_conf[wid] = float(probs[i, preds[i]].item())

        labels = [entry_preds.get(wid, 2) for wid in range(len(entry_tokens))]
        confs = [entry_conf.get(wid, 0.0) for wid in range(len(entry_tokens))]
        return labels, confs

def run_experience_phase1_segment(
    tokens: list[dict],
    resume_id: str = "resume",
    *,
    slug: str | None = None,
) -> dict[str, Any]:
    # 1. EXPERIENCE section tokens — mirror training evaluate.py prep
    exp_section = clean_non_text_tokens([t for t in tokens if t.get("section") == "EXPERIENCE"])
    if not exp_section:
        return _empty_result(tokens, [], resume_id)

    overlay_mongo_field_labels(exp_section, resume_id, slug)
    spatial_all = _normalize_spatial(exp_section)
    spatial_by_token = {id(t): spat for t, spat in zip(exp_section, spatial_all)}

    filtered_tokens = [
        t for t in exp_section
        if t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]
    filtered_tokens.sort(
        key=lambda t: (t.get("page", 0), t.get("lineIndex", 0), t.get("tokenIndex", 0), t.get("x0", 0)),
    )

    for t in filtered_tokens:
        spat = spatial_by_token.get(id(t), [0.0] * 12)
        s = list(spat)
        if len(s) < 12:
            s = s + [0.0] * (12 - len(s))
        t["_temp_spatial"] = s

    filtered_indices = [i for i, t in enumerate(tokens) if t in filtered_tokens]
    # Entry blocks — match training evaluate.py (Mongo entry heads, else B-ENTRY title lines).
    b_entry_lines = resolve_segmentation_entry_heads(filtered_tokens, resume_id, slug)
    if not b_entry_lines:
        import logging
        logging.getLogger(__name__).warning(
            "experience_phase1_segment: no primary entry slice heads — using main-bullet B-ENTRY lines",
        )
        b_entry_lines = {
            (t.get("page"), t.get("lineIndex", t.get("line_index", 0)))
            for t in filtered_tokens
            if (t.get("token") or "").strip() in {"•", "●"}
        }

    # Group tokens into job entries at primary divider lines
    sorted_heads = sorted(list(b_entry_lines))
    
    entries = []
    current_block = []
    current_head = None
    
    for t in filtered_tokens:
        key = (t["page"], t["lineIndex"])
        if key in b_entry_lines and key != current_head:
            if current_block:
                entries.append((current_head, current_block))
                current_block = []
            current_head = key
        if current_head is not None:
            current_block.append(t)
            
    if current_block:
        entries.append((current_head, current_block))

    predictor = get_predictor("experience_phase1_segment", PyTorchExperiencePhase2Predictor)
    
    label_map = ["O", "B-SEG", "I-SEG"]
    
    for head, entry_toks in entries:
        if len(entry_toks) < 3:
            for t in entry_toks:
                t["segLabel"] = "O"
            continue

        is_desc_seq = [False] * len(entry_toks)
        first_desc = -1
        last_desc = -1
        for idx, t in enumerate(entry_toks):
            bio = t.get("_fieldBioLabel") or t.get("bioLabel", "O")
            if bio in ("B-DESC", "I-DESC"):
                if first_desc == -1:
                    first_desc = idx
                last_desc = idx
        if first_desc != -1:
            for idx in range(first_desc, last_desc + 1):
                is_desc_seq[idx] = True

        for idx, t in enumerate(entry_toks):
            if "_temp_spatial" in t:
                spat = list(t["_temp_spatial"])
                if len(spat) < 12:
                    spat = spat + [0.0] * (12 - len(spat))
                spat[11] = float(is_desc_seq[idx])
                t["_temp_spatial"] = spat

        entry_preds_list, entry_conf_list = predictor.segment_entry(entry_toks)
        for t, conf in zip(entry_toks, entry_conf_list):
            t["_segConfidence"] = conf

        # Sequence Continuity Guard
        for idx in range(1, len(entry_toks)):
            t_prev = entry_toks[idx - 1]
            t_curr = entry_toks[idx]
            same_line = (t_curr.get("page") == t_prev.get("page") and t_curr.get("lineIndex") == t_prev.get("lineIndex"))
            is_text = bool(re.search(r'[a-zA-Z0-9]', t_curr.get("token", ""))) and bool(re.search(r'[a-zA-Z0-9]', t_prev.get("token", "")))
            if same_line and is_text and entry_preds_list[idx] == 1 and entry_preds_list[idx - 1] in (1, 2):
                entry_preds_list[idx] = 2
                
        desc_skip = {idx for idx, flag in enumerate(is_desc_seq) if flag}
        entry_preds_list = apply_segment_postprocess(
            entry_toks, entry_preds_list, skip_indices=desc_skip, use_bio_hints=True
        )

        # Generic, model-first BIO repair: keeps the model's own boundaries and
        # only enforces resume-agnostic invariants (separators -> O, entry/after-
        # separator starts a segment, valid BIO). No per-resume tuning.
        entry_preds_list = apply_structural_segmentation(entry_toks, entry_preds_list)

        # Assign back
        for wid, pred in enumerate(entry_preds_list):
            entry_toks[wid]["segLabel"] = label_map[pred]
            
    # Clean temporary spatial properties
    for t in filtered_tokens:
        t.pop("_temp_spatial", None)
        
    # Sync segLabel prediction back to tokens prediction/segLabel (do not overwrite bioLabel!)
    for t in filtered_tokens:
        # Default to O if not processed
        seg = t.get("segLabel", "O")
        # Save prediction in prediction field for ArtifactJsonLoader
        t["prediction"] = seg
        t["segLabel"] = seg
        t["segConfidence"] = round(float(t.pop("_segConfidence", 0.0)), 4)
        
    non_o_count = sum(1 for t in filtered_tokens if t.get("prediction") != "O")
    
    return {
        "stage": "experience_phase1_segment",
        "title": "Experience Token Segmentation",
        "section": "EXPERIENCE",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "experience/phase1_token_segmentation",
        "task": "token_segmentation",
        "phraseSegmentationPipeline": "experience/phase1_token_segmentation",
        "labelField": "prediction",
        "labels": label_map,
        "tokenCount": len(filtered_tokens),
        "evalTokenCount": len(filtered_tokens),
        "nonOCount": non_o_count,
        "sampleLabels": sorted(list(set(t.get("prediction", "O") for t in filtered_tokens))),
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
        ]
    }

def _empty_result(tokens: list[dict], filtered_indices: list[int], resume_id: str) -> dict[str, Any]:
    return {
        "stage": "experience_phase1_segment",
        "title": "Experience Token Segmentation",
        "section": "EXPERIENCE",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "experience/phase1_token_segmentation",
        "task": "token_segmentation",
        "phraseSegmentationPipeline": "experience/phase1_token_segmentation",
        "labelField": "prediction",
        "labels": ["O", "B-SEG", "I-SEG"],
        "tokenCount": 0,
        "evalTokenCount": 0,
        "nonOCount": 0,
        "sampleLabels": [],
        "tokens": []
    }
