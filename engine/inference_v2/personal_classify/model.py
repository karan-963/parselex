"""PersonalSegmentClassifierModel — mirrors training_pipeline/personal/model.py."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .config import GLU_HIDDEN_SIZE, MODEL_NAME, NUM_LABELS, SPATIAL_DIM


class GLUSpatialFusion(nn.Module):
    def __init__(self, text_dim: int, spatial_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.spatial_proj = nn.Linear(spatial_dim, hidden_size)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, text_emb: torch.Tensor, spatial_feat: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text_emb)
        s = self.spatial_proj(spatial_feat)
        gate = torch.sigmoid(self.gate(torch.cat([t, s], dim=-1)))
        return gate * t + (1.0 - gate) * s


class PersonalSegmentClassifierModel(nn.Module):
    def __init__(
        self,
        num_labels: int = NUM_LABELS,
        hidden_size: int = GLU_HIDDEN_SIZE,
        spatial_dim: int = SPATIAL_DIM,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        backbone = model_name or MODEL_NAME
        self.encoder = AutoModel.from_config(
            AutoConfig.from_pretrained(backbone, local_files_only=True)
        )
        self._text_dim = self.encoder.config.hidden_size
        self.spatial_conv = nn.Conv1d(in_channels=spatial_dim, out_channels=64, kernel_size=5, padding=2)
        self.glu_fusion = GLUSpatialFusion(text_dim=self._text_dim, spatial_dim=64, hidden_size=hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_segs, seg_len = input_ids.shape
        text_dim = self._text_dim
        flat_input_ids = input_ids.view(batch_size * num_segs, seg_len)
        flat_attn_mask = attention_mask.view(batch_size * num_segs, seg_len)

        if torch.onnx.is_in_onnx_export():
            enc_out = self.encoder(input_ids=flat_input_ids, attention_mask=flat_attn_mask)
            token_embeddings = enc_out.last_hidden_state
            mask_expanded = flat_attn_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask_expanded, dim=1)
            denom = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            text_embeddings = (summed / denom).view(batch_size, num_segs, text_dim)
        else:
            non_pad_mask = flat_attn_mask.sum(dim=-1) > 0
            non_pad_indices = non_pad_mask.nonzero(as_tuple=False).squeeze(-1)
            flat_text_embeddings = torch.zeros(
                (batch_size * num_segs, text_dim),
                device=input_ids.device,
                dtype=torch.float32,
            )
            if len(non_pad_indices) > 0:
                enc_out = self.encoder(
                    input_ids=flat_input_ids[non_pad_indices],
                    attention_mask=flat_attn_mask[non_pad_indices],
                )
                token_embeddings = enc_out.last_hidden_state
                mask_expanded = flat_attn_mask[non_pad_indices].unsqueeze(-1).expand(token_embeddings.size()).float()
                active_sum = torch.sum(token_embeddings * mask_expanded, dim=1)
                active_denom = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                flat_text_embeddings[non_pad_indices] = (active_sum / active_denom).to(flat_text_embeddings.dtype)
            text_embeddings = flat_text_embeddings.view(batch_size, num_segs, text_dim)

        spatial_in = spatial_features.permute(0, 2, 1)
        spatial_conv_out = torch.relu(self.spatial_conv(spatial_in))
        spatial_embeddings = spatial_conv_out.permute(0, 2, 1)
        fused = self.dropout(self.glu_fusion(text_embeddings, spatial_embeddings))
        return self.classifier(fused)
