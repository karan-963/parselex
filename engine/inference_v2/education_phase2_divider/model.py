"""Phrase classification boundary model (MiniLM + 19D spatial GLU)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .config import SPATIAL_DIM, MODEL_NAME


class GLUSpatialFusion(nn.Module):
    def __init__(self, text_dim: int, spatial_dim: int, hidden_size: int):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.spatial_proj = nn.Linear(spatial_dim, hidden_size)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, text_emb: torch.Tensor, spatial_feat: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text_emb)
        s = self.spatial_proj(spatial_feat)
        gate = torch.sigmoid(self.gate(torch.cat([t, s], dim=-1)))
        return gate * t + (1.0 - gate) * s


class PhraseSectionDividerModel(nn.Module):
    def __init__(self, num_labels: int = 3, hidden_size: int = 256, model_name: str | None = None):
        super().__init__()
        backbone = model_name or MODEL_NAME
        self.encoder = AutoModel.from_config(
            AutoConfig.from_pretrained(backbone, local_files_only=True)
        )
        text_dim = self.encoder.config.hidden_size

        self.spatial_conv = nn.Conv1d(
            in_channels=SPATIAL_DIM,
            out_channels=64,
            kernel_size=5,
            padding=2,
        )
        self.glu_fusion = GLUSpatialFusion(text_dim=text_dim, spatial_dim=64, hidden_size=hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, spatial_features: torch.Tensor):
        b, n, length = input_ids.shape
        text_dim = self.encoder.config.hidden_size
        flat_input_ids = input_ids.view(b * n, length)
        flat_attn_mask = attention_mask.view(b * n, length)

        non_pad_mask = flat_attn_mask.sum(dim=-1) > 0
        non_pad_indices = non_pad_mask.nonzero().squeeze(-1)
        flat_text_embeddings = torch.zeros((b * n, text_dim), device=input_ids.device, dtype=torch.float32)

        if len(non_pad_indices) > 0:
            enc_out = self.encoder(
                input_ids=flat_input_ids[non_pad_indices],
                attention_mask=flat_attn_mask[non_pad_indices],
            )
            token_embeddings = enc_out.last_hidden_state
            active_mask = flat_attn_mask[non_pad_indices].unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * active_mask, dim=1)
            denom = torch.clamp(active_mask.sum(dim=1), min=1e-9)
            flat_text_embeddings[non_pad_indices] = (summed / denom).to(flat_text_embeddings.dtype)

        text_embeddings = flat_text_embeddings.view(b, n, text_dim)
        spatial_in = spatial_features.permute(0, 2, 1)
        spatial_out = torch.relu(self.spatial_conv(spatial_in)).permute(0, 2, 1)
        fused = self.dropout(self.glu_fusion(text_embeddings, spatial_out))
        return self.classifier(fused)


def build_segmenter(*, num_labels: int, model_name: str) -> PhraseSectionDividerModel:
    return PhraseSectionDividerModel(num_labels=num_labels, model_name=model_name)
