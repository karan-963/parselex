from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel
from .base import GLUSpatialFusion

SPATIAL_DIM     = 10
BACKBONE_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
DROPOUT         = 0.1
CONTEXT_EMB_DIM = 32

class ResumeSectionClassifier(nn.Module):
    """Phase 2: sequence classifier over a heading+content chunk.

    spatial_features is (B, spatial_dim) — heading token's spatial vector.
    Uses the [CLS] embedding fused with heading spatial for classification.
    """

    def __init__(self, num_labels: int, spatial_dim: int = SPATIAL_DIM, model_name: str = BACKBONE_NAME, dropout: float = DROPOUT):
        super().__init__()
        self.spatial_dim    = spatial_dim
        self.encoder        = AutoModel.from_pretrained(model_name)
        h                   = self.encoder.config.hidden_size
        self.spatial_fusion = GLUSpatialFusion(spatial_dim, h)
        self.dropout        = nn.Dropout(dropout)
        self.prev_label_emb = nn.Embedding(num_labels + 1, CONTEXT_EMB_DIM)
        self.classifier     = nn.Linear(h + CONTEXT_EMB_DIM, num_labels)
        self.num_labels     = num_labels

    def forward(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        spatial_features: torch.Tensor,       # (B, spatial_dim) — chunk-level heading spatial
        labels:           torch.Tensor | None = None,
        prev_labels:      torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        # Workaround for PyTorch CPU batch size 1 SIGBUS crash
        is_cpu_single_batch = (input_ids.shape[0] == 1 and input_ids.device.type == "cpu")
        if is_cpu_single_batch:
            input_ids = torch.cat([input_ids, input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
            spatial_features = torch.cat([spatial_features, spatial_features], dim=0)
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)
            if prev_labels is not None:
                prev_labels = torch.cat([prev_labels, prev_labels], dim=0)

        enc     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = enc.last_hidden_state[:, 0, :]          # (B, H)
        h       = self.spatial_fusion(cls_emb, spatial_features)  # (B, H)
        h       = self.dropout(h)
        if prev_labels is not None:
            prev_emb = self.prev_label_emb(prev_labels)          # (B, CONTEXT_EMB_DIM)
        else:
            start = torch.full((h.shape[0],), self.num_labels, dtype=torch.long, device=h.device)
            prev_emb = self.prev_label_emb(start)
        h = torch.cat([h, prev_emb], dim=-1)                     # (B, H + CONTEXT_EMB_DIM)
        logits  = self.classifier(h)                       # (B, num_labels)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        if is_cpu_single_batch:
            if loss is not None:
                loss = loss.mean()
            if logits is not None:
                logits = logits[:1]

        return {"loss": loss, "logits": logits}


class BlockClassifier(nn.Module):
    def __init__(self, num_labels: int, model_name: str = BACKBONE_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h // 2, num_labels)
        )
        self.spatial_proj = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, num_labels)
        )
        self.num_labels = num_labels

    def forward(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        spatial_features: torch.Tensor,
        labels:           torch.Tensor | None = None,
        **kwargs
    ) -> dict[str, torch.Tensor | None]:
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        token_embeddings = enc.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        text_feat = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        logits_text = self.text_proj(text_feat)
        
        # Extract first 4 elements of spatial features: x0, y0, x1, y1
        if spatial_features.dim() == 3:
            spatial_coords = spatial_features[:, 0, :4]
        else:
            spatial_coords = spatial_features[:, :4]
            
        logits_spatial = self.spatial_proj(spatial_coords)
        
        # Late fusion: 10% spatial weight contribution
        logits = 0.9 * logits_text + 0.1 * logits_spatial
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            
        return {
            "logits": logits,
            "loss": loss
        }
