import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class SpatialLayoutProjectionGate(nn.Module):
    """Trunk 2: 10-D spatial layout projection gate.
    Projects 10-D spatial features to 768-D and performs a sigmoid-gated late-fusion.
    """
    def __init__(self, spatial_dim: int = 10, hidden_size: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(spatial_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, text_emb: torch.Tensor, spatial_feat: torch.Tensor) -> torch.Tensor:
        # text_emb is (B, H), spatial_feat is (B, spatial_dim)
        s = self.proj(spatial_feat)  # (B, H)
        gate_input = torch.cat([text_emb, s], dim=-1)  # (B, 2*H)
        g = self.gate(gate_input)  # (B, H)
        return g * text_emb + (1.0 - g) * s  # (B, H)

class ResumeSectionClassifier(nn.Module):
    """Phase 2 Resume Section Classifier with 3 feature trunks:
    1. Text Trunk: 768-D DistilRoBERTa encoder
    2. Spatial Trunk: 10-D Spatial Layout Projection Gate
    3. Context Trunk: Autoregressive 32-D prev_label embedding
    """
    def __init__(self, num_classes: int = 7, spatial_dim: int = 10, model_name: str = "distilroberta-base", dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        
        # Trunk 1: 768-D text encoder backbone
        self.encoder = AutoModel.from_config(
            AutoConfig.from_pretrained(model_name, local_files_only=True)
        )
        h = self.encoder.config.hidden_size  # 768
        
        # Trunk 2: 10-D spatial layout projection gate
        self.spatial_gate = SpatialLayoutProjectionGate(spatial_dim, h)
        
        # Trunk 3: Autoregressive 32-D prev_label embedding block
        self.prev_label_emb = nn.Embedding(num_classes + 1, 32)
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification head combining fused text-spatial features (768-D) + prev_label embedding (32-D)
        self.classifier = nn.Linear(h + 32, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        spatial_features: torch.Tensor,
        prev_labels: torch.Tensor | None = None,
        labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        # Workaround for PyTorch CPU batch size 1 SIGBUS crash
        is_cpu_single_batch = (input_ids.shape[0] == 1 and input_ids.device.type == "cpu")
        if is_cpu_single_batch:
            input_ids = torch.cat([input_ids, input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
            spatial_features = torch.cat([spatial_features, spatial_features], dim=0)
            if prev_labels is not None:
                prev_labels = torch.cat([prev_labels, prev_labels], dim=0)
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)

        # Trunk 1: Text features (CLS token)
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = enc.last_hidden_state[:, 0, :]  # (B, H)
        
        # Trunk 2: Spatial features projected and gated
        fused = self.spatial_gate(cls_emb, spatial_features)  # (B, H)
        fused = self.dropout(fused)
        
        # Trunk 3: Prev label context
        if prev_labels is not None:
            prev_emb = self.prev_label_emb(prev_labels)  # (B, 32)
        else:
            # Fallback to start token index (num_classes)
            start_tokens = torch.full((fused.shape[0],), self.num_classes, dtype=torch.long, device=fused.device)
            prev_emb = self.prev_label_emb(start_tokens)  # (B, 32)
            
        # Concatenate fused text/spatial features (768-D) with context embedding (32-D)
        combined = torch.cat([fused, prev_emb], dim=-1)  # (B, H + 32)
        logits = self.classifier(combined)  # (B, num_classes)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
        if is_cpu_single_batch:
            if loss is not None:
                loss = loss.mean()
            if logits is not None:
                logits = logits[:1]
                
        return {"loss": loss, "logits": logits}
