"""Load SkillsSegmentClassifierModel and run token-level inference."""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer

from inference_v2.model_precision import apply_precision
from .config import MODEL_NAME, NUM_LABELS
from .inference_engine import infer_token_labels
from .model import SkillsSegmentClassifierModel
from .postprocess import postprocess_skill_predictions
from .training_bridge import load_training_helpers


class SkillsPhase7Predictor:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"Skills model not found: {best_model_path}")

        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        num_labels = NUM_LABELS
        if "classifier.weight" in state_dict:
            num_labels = state_dict["classifier.weight"].shape[0]

        self.model = SkillsSegmentClassifierModel(num_labels=num_labels, model_name=MODEL_NAME)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()
        self.model, self.device = apply_precision(self.model, best_model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True, local_files_only=True)
        self._helpers = load_training_helpers()

    def classify_tokens(
        self,
        resume_tokens: list[dict],
        all_tokens: list[dict] | None = None,
    ) -> tuple[list[str], dict[int, float]]:
        raw_labels, token_confs = infer_token_labels(
            resume_tokens,
            self.model,
            self.tokenizer,
            self._helpers,
            self.device,
            all_tokens=all_tokens,
        )
        labels = postprocess_skill_predictions(resume_tokens, raw_labels)
        return labels, token_confs
