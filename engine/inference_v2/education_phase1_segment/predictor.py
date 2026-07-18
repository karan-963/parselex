"""Education phase 1 segmenter predictor."""

from __future__ import annotations

import os

import torch
from transformers import AutoTokenizer

from inference_v2.confidence import word_level_confidences

from inference_v2.model_precision import apply_precision

from .config import MODEL_NAME, NUM_LABELS
from .model import build_segmenter


class EducationPhase1Predictor:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )

        best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"Education phase1 segmenter model not found: {best_model_path}")

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
            dtype=torch.float32, device=self.device,
        )
        y0 = torch.tensor(
            [entry_spatial[wid][1] if (wid is not None and wid < len(entry_spatial)) else 0.0 for wid in word_ids],
            dtype=torch.float32, device=self.device,
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

        entry_preds: dict[int, int] = {}
        for i, wid in enumerate(word_ids):
            if wid is not None and wid < len(entry_tokens) and wid not in entry_preds:
                entry_preds[wid] = int(preds[i])

        labels = [entry_preds.get(wid, 2) for wid in range(len(entry_tokens))]
        confs = word_level_confidences(logits[0], word_ids, len(entry_tokens))
        return labels, confs
