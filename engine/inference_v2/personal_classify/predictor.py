"""Load PersonalSegmentClassifierModel and run personal segment inference."""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer

from inference_v2.confidence import batch_max_probs
from inference_v2.model_precision import apply_precision
from .config import ID2LABEL, LABEL_LIST, MAX_SEG_LEN, MAX_SEGS, MODEL_NAME, NUM_LABELS, SPATIAL_DIM
from .model import PersonalSegmentClassifierModel
from .training_bridge import load_training_helpers


def _resolve_checkpoint_path() -> str:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(module_dir, "best_model.pt")
    if os.path.isfile(local_path):
        return local_path
    raise FileNotFoundError(f"Personal model not found: {local_path}")


class PersonalPhase15Predictor:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        checkpoint_path = _resolve_checkpoint_path()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        num_labels = NUM_LABELS
        if "classifier.weight" in state_dict:
            num_labels = state_dict["classifier.weight"].shape[0]

        self.model = PersonalSegmentClassifierModel(num_labels=num_labels, model_name=MODEL_NAME)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()
        self.model, self.device = apply_precision(self.model, checkpoint_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True, local_files_only=True)
        self._helpers = load_training_helpers()
        self._id2label = {i: ID2LABEL.get(i, LABEL_LIST[i]) for i in range(num_labels) if i < len(LABEL_LIST)}

    @torch.no_grad()
    def classify_personal_segments(
        self,
        personal_segments: list[dict],
        all_cleaned_tokens: list[dict],
    ) -> tuple[list[str], list[dict], list[float]]:
        if not personal_segments:
            return [], [], []

        helpers = self._helpers
        spatial_features = [
            helpers["extract_segment_spatial"](segment, all_tokens=all_cleaned_tokens, spatial_dim=SPATIAL_DIM)
            for segment in personal_segments
        ]

        seg_texts = [segment.get("text", "") for segment in personal_segments]
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
            pad_tok = torch.full((MAX_SEG_LEN,), self.tokenizer.pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros(MAX_SEG_LEN, dtype=torch.long)
            for _ in range(MAX_SEGS - num_segs):
                seg_input_ids.append(pad_tok.clone())
                seg_attn_mask.append(pad_mask.clone())
                spatial_features.append([0.0] * SPATIAL_DIM)
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
        pred_labels = [self._id2label.get(prediction, "O") for prediction in preds]
        active_segments = personal_segments[:num_segs]
        pred_labels = helpers["post_process_segment_predictions"](active_segments, pred_labels)
        return pred_labels, active_segments, pred_confs
