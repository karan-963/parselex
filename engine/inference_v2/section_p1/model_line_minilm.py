"""MiniLM line heading classifier with spatial MLP side channel."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from . import config


class LineHeadingMiniLM(nn.Module):
    def __init__(
        self,
        backbone: str = config.LINE_MINILM_NAME,
        spatial_dim: int = config.LINE_SPATIAL_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_config(
            AutoConfig.from_pretrained(backbone, local_files_only=True)
        )
        h = self.encoder.config.hidden_size
        self.spatial_mlp = nn.Sequential(
            nn.Linear(spatial_dim, h // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h // 4, h // 4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h + h // 4, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        spatial: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        spat = self.spatial_mlp(spatial)
        logits = self.classifier(torch.cat([pooled, spat], dim=-1))

        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result
