"""Load PhraseSegmentClassifierModel and run segment-level inference."""

from __future__ import annotations

import os
from collections import Counter

import torch
from transformers import AutoTokenizer

from inference_v2.confidence import batch_max_probs
from inference_v2.model_precision import apply_precision
from .config import MAX_SEG_LEN, MAX_SEGS, MODEL_ID2LABEL, MODEL_LABEL_LIST, MODEL_NAME
from .date_resolve import resolve_dates_to_sdate_edate
from .model import PhraseSegmentClassifierModel
from .training_bridge import load_training_helpers


class ProjectPhase3Predictor:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"Project phase 3 model not found: {best_model_path}")

        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        num_labels = len(MODEL_LABEL_LIST)
        head_key = "classifier.weight"
        if head_key in state_dict:
            num_labels = state_dict[head_key].shape[0]

        self.model = PhraseSegmentClassifierModel(num_labels=num_labels, model_name=MODEL_NAME)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()
        self.model, self.device = apply_precision(self.model, best_model_path, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True, local_files_only=True)
        self._helpers = load_training_helpers()

    def _project_font_brackets(self, project_segments: list[dict]) -> tuple[float, float, float]:
        sizes = [
            s["spatial"][0]
            for s in project_segments
            if s.get("spatial") and len(s["spatial"]) > 0
        ]
        if not sizes:
            sizes = [10.0]
        max_size = max(sizes)
        min_size = min(sizes)
        default_size = Counter(sizes).most_common(1)[0][0]
        return max_size, default_size, min_size

    @torch.no_grad()
    def classify_segments(self, tokens: list[dict]) -> tuple[list[dict], list[str], list[dict], list[float]]:
        """Build segments from tokens, classify project segments, return resolved labels."""
        h = self._helpers
        cleaned = h["clean_cid_tokens"](tokens)
        segments = h["construct_sentences_by_appearance"](cleaned)
        segments = h["split_hyphenated_segments"](segments)
        if not segments:
            return [], [], [], []

        project_segments: list[dict] = []
        for s in segments:
            if h["is_project_segment"](s):
                project_segments.append(s)

        if not project_segments:
            return segments, [], [], []

        max_size, default_size, min_size = self._project_font_brackets(project_segments)
        spatial_matrix = h["build_spatial_feature_matrix"](segments, max_size, default_size, min_size)
        spatial_features = [spatial_matrix[i] for i, s in enumerate(segments) if h["is_project_segment"](s)]

        seg_texts = [s["text"] for s in project_segments]
        seg_input_ids: list[torch.Tensor] = []
        seg_attn_mask: list[torch.Tensor] = []

        for text in seg_texts[:MAX_SEGS]:
            enc = self.tokenizer(
                text,
                max_length=MAX_SEG_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            seg_input_ids.append(enc["input_ids"].squeeze(0))
            seg_attn_mask.append(enc["attention_mask"].squeeze(0))

        num_segs = len(seg_texts)
        if num_segs < MAX_SEGS:
            for _ in range(MAX_SEGS - num_segs):
                seg_input_ids.append(torch.full((MAX_SEG_LEN,), self.tokenizer.pad_token_id, dtype=torch.long))
                seg_attn_mask.append(torch.zeros(MAX_SEG_LEN, dtype=torch.long))
                spatial_features.append([0.0] * 16)
        else:
            seg_input_ids = seg_input_ids[:MAX_SEGS]
            seg_attn_mask = seg_attn_mask[:MAX_SEGS]
            spatial_features = spatial_features[:MAX_SEGS]
            num_segs = MAX_SEGS

        input_ids = torch.stack(seg_input_ids).unsqueeze(0).to(self.device)
        attention_mask = torch.stack(seg_attn_mask).unsqueeze(0).to(self.device)
        spatial_t = torch.tensor(spatial_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        logits = self.model(input_ids, attention_mask, spatial_t)
        preds = logits.squeeze(0).argmax(-1).cpu().tolist()[:num_segs]
        pred_confs = batch_max_probs(logits.squeeze(0)[:num_segs], [int(p) for p in preds])

        pred_labels_4d = [MODEL_ID2LABEL[p] for p in preds]
        pred_labels_4d = h["postprocess_segment_predictions"](project_segments[:num_segs], pred_labels_4d)
        resolved = resolve_dates_to_sdate_edate(pred_labels_4d, project_segments[:num_segs])

        return segments, resolved, project_segments[:num_segs], pred_confs
