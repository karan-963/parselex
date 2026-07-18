"""PhraseSegmentClassifierModel — mirrors training phase3 model.py."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .config import MODEL_NAME, SPATIAL_DIM


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


class PhraseSegmentClassifierModel(nn.Module):
    def __init__(self, num_labels: int = 4, hidden_size: int = 256, model_name: str | None = None):
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
        self.segment_bigru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.body_gate = nn.Linear(hidden_size, 1)
        self.fp_suppress = nn.Parameter(torch.tensor(2.5))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, spatial_features: torch.Tensor):
        B, N, L = input_ids.shape
        text_dim = self.encoder.config.hidden_size

        flat_input_ids = input_ids.view(B * N, L)
        flat_attn_mask = attention_mask.view(B * N, L)

        non_pad_mask = flat_attn_mask.sum(dim=-1) > 0
        non_pad_indices = non_pad_mask.nonzero().squeeze(-1)
        flat_text_embeddings = torch.zeros((B * N, text_dim), device=input_ids.device, dtype=torch.float32)

        if len(non_pad_indices) > 0:
            active_input_ids = flat_input_ids[non_pad_indices]
            active_attn_mask = flat_attn_mask[non_pad_indices]
            enc_out = self.encoder(input_ids=active_input_ids, attention_mask=active_attn_mask)
            token_embeddings = enc_out.last_hidden_state
            active_mask_expanded = active_attn_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            active_sum_embeddings = torch.sum(token_embeddings * active_mask_expanded, dim=1)
            active_sum_mask = torch.clamp(active_mask_expanded.sum(dim=1), min=1e-9)
            flat_text_embeddings[non_pad_indices] = (active_sum_embeddings / active_sum_mask).to(
                flat_text_embeddings.dtype
            )

        text_embeddings = flat_text_embeddings.view(B, N, text_dim)
        spatial_in = spatial_features.permute(0, 2, 1)
        spatial_conv_out = torch.relu(self.spatial_conv(spatial_in))
        spatial_embeddings = spatial_conv_out.permute(0, 2, 1)

        fused = self.glu_fusion(text_embeddings, spatial_embeddings)
        seg_mask = (attention_mask.sum(dim=-1) > 0).unsqueeze(-1).to(fused.dtype)
        fused = fused * seg_mask
        contextual, _ = self.segment_bigru(fused)
        contextual = contextual * seg_mask
        contextual = self.dropout(contextual)

        body_logits = self.body_gate(contextual).squeeze(-1)
        logits = self.classifier(contextual)
        body_score = torch.sigmoid(body_logits)
        logits = logits.clone()
        logits[..., 1] = logits[..., 1] - self.fp_suppress * body_score
        return logits
