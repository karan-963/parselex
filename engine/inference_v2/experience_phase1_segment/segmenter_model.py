"""Phrase segmenter architecture for experience phase-1 token segmentation.

Mirrors ``training_pipeline/experience/phase1_token_segmentation/src/model.py`` so
the exported ``best_model.pt`` loads with fully-matching weights: distilled encoder
→ ``text_projection`` → GLU spatial fusion → conv ``boundary_gate`` emission head.

Kept local to this module (strict section isolation) because the shared
``models.token_model`` architecture diverged (coordinate embeddings + attention
spatial-bias) and is incompatible with this checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class GLUSpatialFusion(nn.Module):
    """Gated Linear Unit for late spatial-text feature integration."""

    def __init__(self, spatial_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(spatial_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, text_emb: torch.Tensor, spatial_feat: torch.Tensor) -> torch.Tensor:
        s = self.spatial_mlp(spatial_feat)
        gate = torch.sigmoid(self.gate(torch.cat([text_emb, s], dim=-1)))
        return gate * text_emb + (1 - gate) * s


class SpatialBoundaryGate(nn.Module):
    """Deep early-fusion emission trunk (1D conv over 5 adjacent tokens)."""

    def __init__(self, spatial_in_dim: int = 12, num_labels: int = 3, hidden_size: int = 768) -> None:
        super().__init__()
        self.spatial_conv = nn.Conv1d(
            in_channels=spatial_in_dim,
            out_channels=128,
            kernel_size=5,
            padding=2,
        )
        self.classifier_head = nn.Linear(hidden_size + 128, num_labels)

    def forward(self, hidden_states: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        x = spatial_features.permute(0, 2, 1)
        x = torch.relu(self.spatial_conv(x))
        spatial_up_emb = x.permute(0, 2, 1)
        fused = torch.cat([hidden_states, spatial_up_emb], dim=-1)
        return self.classifier_head(fused)


class PhraseSegmenterTransformer(nn.Module):
    def __init__(
        self,
        num_labels: int = 3,
        spatial_dim: int = 12,
        dropout: float = 0.1,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_crf: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.num_labels = num_labels
        self.use_crf = use_crf

        self.encoder = AutoModel.from_config(
            AutoConfig.from_pretrained(model_name, local_files_only=True),
            attn_implementation="eager",
        )
        text_hidden_dim = self.encoder.config.hidden_size

        self.text_projection = nn.Linear(text_hidden_dim, text_hidden_dim)
        self.spatial_fusion = GLUSpatialFusion(spatial_dim, text_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.boundary_gate = SpatialBoundaryGate(
            spatial_in_dim=spatial_dim, num_labels=num_labels, hidden_size=text_hidden_dim
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        spatial_features: torch.Tensor,
        spatial_matrix: torch.Tensor | None = None,  # unused; kept for call-site parity
        labels: torch.Tensor | None = None,
    ):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_h = enc.last_hidden_state
        projected_text = self.text_projection(text_h)
        fused = self.spatial_fusion(projected_text, spatial_features)
        fused = self.dropout(fused)
        logits = self.boundary_gate(fused, spatial_features)

        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.num_labels), labels.view(-1)
            )
            return logits, loss
        return logits
