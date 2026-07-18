"""Phrase classification boundary model wrapper."""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from .config import SPATIAL_DIM, MODEL_NAME

class GLUSpatialFusion(nn.Module):
    """Gated Linear Unit for late spatial-text feature integration."""
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
        
        # Track 2: 1D Spatial Convolution for layout and visual properties
        self.spatial_conv = nn.Conv1d(
            in_channels=SPATIAL_DIM,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        
        # Fusion: Gated Linear Unit (GLU)
        self.glu_fusion = GLUSpatialFusion(
            text_dim=text_dim,
            spatial_dim=64,
            hidden_size=hidden_size
        )
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, spatial_features: torch.Tensor):
        # input_ids: (B, N, L_text)
        # attention_mask: (B, N, L_text)
        # spatial_features: (B, N, SPATIAL_DIM)
        
        B, N, L = input_ids.shape
        text_dim = self.encoder.config.hidden_size
        
        # Track 1: MiniLM token representation
        flat_input_ids = input_ids.view(B * N, L)
        flat_attn_mask = attention_mask.view(B * N, L)
        
        # Optimize memory and speed: only run the transformer encoder on non-padded segments
        # Padded segments have attention mask sum equal to 0
        non_pad_mask = flat_attn_mask.sum(dim=-1) > 0
        non_pad_indices = non_pad_mask.nonzero().squeeze(-1)
        
        flat_text_embeddings = torch.zeros((B * N, text_dim), device=input_ids.device, dtype=torch.float32)
        
        if len(non_pad_indices) > 0:
            active_input_ids = flat_input_ids[non_pad_indices]
            active_attn_mask = flat_attn_mask[non_pad_indices]
            
            enc_out = self.encoder(input_ids=active_input_ids, attention_mask=active_attn_mask)
            token_embeddings = enc_out.last_hidden_state  # (Active_B, L, text_dim)
            
            # Mean Pooling over active text length
            active_mask_expanded = active_attn_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            active_sum_embeddings = torch.sum(token_embeddings * active_mask_expanded, dim=1)
            active_sum_mask = torch.clamp(active_mask_expanded.sum(dim=1), min=1e-9)
            active_embeddings = active_sum_embeddings / active_sum_mask
            
            # Scatter back into pre-allocated flat_text_embeddings
            flat_text_embeddings[non_pad_indices] = active_embeddings.to(flat_text_embeddings.dtype)
            
        text_embeddings = flat_text_embeddings.view(B, N, text_dim) # (B, N, 384)
        
        # Track 2: 1D spatial convolution over the sequence dimension N
        spatial_in = spatial_features.permute(0, 2, 1) # (B, SPATIAL_DIM, N)
        spatial_conv_out = torch.relu(self.spatial_conv(spatial_in)) # (B, 64, N)
        spatial_embeddings = spatial_conv_out.permute(0, 2, 1) # (B, N, 64)
        
        # GLU Fusion Integration
        fused = self.glu_fusion(text_embeddings, spatial_embeddings) # (B, N, hidden_size)
        fused = self.dropout(fused)
        
        # Classification Logits
        logits = self.classifier(fused) # (B, N, num_labels)
        
        return logits

def build_segmenter(*, num_labels: int, spatial_dim: int, model_name: str) -> PhraseSectionDividerModel:
    return PhraseSectionDividerModel(
        num_labels=num_labels,
        model_name=model_name,
    )
