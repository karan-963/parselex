from __future__ import annotations
import types
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from .base import GLUSpatialFusion, custom_attention_forward, custom_segmenter_attention_forward

SPATIAL_DIM     = 16
BACKBONE_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
DROPOUT         = 0.1

class ResumeTokenClassifier(nn.Module):
    def __init__(self, num_labels: int, spatial_dim: int = SPATIAL_DIM, model_name: str = BACKBONE_NAME, dropout: float = DROPOUT, use_crf: bool = False, is_bio: bool = False, id2lab: dict[int, str] | None = None):
        super().__init__()
        self.spatial_dim    = spatial_dim
        # Force eager attention implementation so that we can patch self-attention Layer forward passes
        self.encoder        = AutoModel.from_config(
            AutoConfig.from_pretrained(model_name, local_files_only=True),
            attn_implementation="eager",
        )
        h                   = self.encoder.config.hidden_size
        
        # 4 Coordinate embeddings for early spatial bucketing (x0, y0, x1, y1)
        self.x0_emb = nn.Embedding(1001, h)
        self.y0_emb = nn.Embedding(1001, h)
        self.x1_emb = nn.Embedding(1001, h)
        self.y1_emb = nn.Embedding(1001, h)
        
        self.spatial_fusion = GLUSpatialFusion(spatial_dim, h)
        self.dropout        = nn.Dropout(dropout)
        self.classifier     = nn.Linear(h, num_labels)
        self.num_labels     = num_labels
        self.use_crf        = use_crf
        self.is_bio         = is_bio
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_labels, batch_first=True)
            self._initialize_crf_transitions(id2lab)
            
        self._patch_encoder_attention_bias()

    def _patch_encoder_attention_bias(self):
        # Identify the self-attention layers to patch: final 3 layers (3, 4, 5)
        # DistilRoBERTa has 6 layers under self.encoder.encoder.layer
        for idx in [3, 4, 5]:
            if idx < len(self.encoder.encoder.layer):
                self_attn = self.encoder.encoder.layer[idx].attention.self
                
                # Check if already patched to avoid multiple additions
                if not hasattr(self_attn, "spatial_bias_proj"):
                    # Create projection layer: 2 input channels (dx, dy) -> num_attention_heads outputs
                    proj = nn.Linear(2, self_attn.num_attention_heads)
                    self_attn.add_module("spatial_bias_proj", proj)
                    
                    # Store a reference to the encoder/parent to access temporary spatial_matrix
                    self_attn.__dict__["encoder_ref"] = self.encoder
                    
                    # Bind the custom attention forward method
                    self_attn.forward = types.MethodType(custom_attention_forward, self_attn)

    def _initialize_crf_transitions(self, id2lab: dict[int, str] | None = None):
        """Programmatically forbid invalid BIO/BILOU transitions."""
        invalid_mask = torch.zeros(self.num_labels, self.num_labels, dtype=torch.bool)
        invalid_start_mask = torch.zeros(self.num_labels, dtype=torch.bool)
        
        if id2lab is not None:
            lab2id = {v: k for k, v in id2lab.items()}
            for to_idx in range(self.num_labels):
                to_label = id2lab.get(to_idx, "O")
                if to_label.startswith("I-"):
                    tag = to_label[2:]
                    b_label = "B-" + tag
                    b_idx = lab2id.get(b_label)
                    
                    invalid_start_mask[to_idx] = True
                    for from_idx in range(self.num_labels):
                        if from_idx != b_idx and from_idx != to_idx:
                            invalid_mask[from_idx, to_idx] = True
                elif to_label.startswith("L-"):
                    tag = to_label[2:]
                    b_label = "B-" + tag
                    i_label = "I-" + tag
                    b_idx = lab2id.get(b_label)
                    i_idx = lab2id.get(i_label)
                    
                    invalid_start_mask[to_idx] = True
                    for from_idx in range(self.num_labels):
                        if from_idx != b_idx and from_idx != i_idx:
                            invalid_mask[from_idx, to_idx] = True
        else:
            if self.is_bio:
                # BIO scheme
                num_tags = (self.num_labels - 1) // 2
                for k in range(1, num_tags + 1):
                    b_idx = 2 * k - 1
                    i_idx = 2 * k
                    
                    # Starting with I-tag is forbidden
                    invalid_start_mask[i_idx] = True
                    
                    # Transition to I-tag (i_idx) must come from B-tag (b_idx) or I-tag (i_idx) of the same type
                    for from_idx in range(self.num_labels):
                        if from_idx != b_idx and from_idx != i_idx:
                            invalid_mask[from_idx, i_idx] = True
            else:
                # BILOU scheme
                num_tags = (self.num_labels - 1) // 4
                for k in range(1, num_tags + 1):
                    b_idx = 4 * k - 3
                    i_idx = 4 * k - 2
                    l_idx = 4 * k - 1
                    u_idx = 4 * k
                    
                    # Starting with I-tag or L-tag is forbidden
                    invalid_start_mask[i_idx] = True
                    invalid_start_mask[l_idx] = True
                    
                    # Transition to I-tag must come from B-tag or I-tag of same type
                    for from_idx in range(self.num_labels):
                        if from_idx != b_idx and from_idx != i_idx:
                            invalid_mask[from_idx, i_idx] = True
                            
                    # Transition to L-tag must come from B-tag or I-tag of same type
                    for from_idx in range(self.num_labels):
                        if from_idx != b_idx and from_idx != i_idx:
                            invalid_mask[from_idx, l_idx] = True
                            
                    # Transition from B-tag must go to I-tag or L-tag of same type
                    for to_idx in range(self.num_labels):
                        if to_idx != i_idx and to_idx != l_idx:
                            invalid_mask[b_idx, to_idx] = True
                            
                    # Transition from I-tag must go to I-tag or L-tag of same type
                    for to_idx in range(self.num_labels):
                        if to_idx != i_idx and to_idx != l_idx:
                            invalid_mask[i_idx, to_idx] = True

        # Apply masks initially
        self.register_buffer("invalid_mask", invalid_mask)
        self.register_buffer("invalid_start_mask", invalid_start_mask)

    def enforce_hard_constraints(self):
        """Enforce static, hard-coded transitions and start constraints, overriding weight decay."""
        if not self.use_crf:
            return
        with torch.no_grad():
            self.crf.transitions.copy_(
                torch.where(
                    self.invalid_mask.to(self.crf.transitions.device),
                    torch.tensor(-10000.0, device=self.crf.transitions.device),
                    self.crf.transitions
                )
            )
            self.crf.start_transitions.copy_(
                torch.where(
                    self.invalid_start_mask.to(self.crf.start_transitions.device),
                    torch.tensor(-10000.0, device=self.crf.start_transitions.device),
                    self.crf.start_transitions
                )
            )

    def forward(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        spatial_features: torch.Tensor,
        labels:           torch.Tensor | None = None,
        spatial_matrix:   torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        # Workaround for PyTorch CPU batch size 1 SIGBUS crash
        is_cpu_single_batch = (input_ids.shape[0] == 1 and input_ids.device.type == "cpu")
        if is_cpu_single_batch:
            input_ids = torch.cat([input_ids, input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
            spatial_features = torch.cat([spatial_features, spatial_features], dim=0)
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)
            if spatial_matrix is not None:
                spatial_matrix = torch.cat([spatial_matrix, spatial_matrix], dim=0)

        # Extract early layout coordinate features directly from the first 4 components of 10D/11D spatial_features (x0n, y0n, x1n, y1n)
        x0 = torch.clamp(spatial_features[..., 0], 0.0, 1.0)
        y0 = torch.clamp(spatial_features[..., 1], 0.0, 1.0)
        x1 = torch.clamp(spatial_features[..., 2], 0.0, 1.0)
        y1 = torch.clamp(spatial_features[..., 3], 0.0, 1.0)

        # Store spatial_matrix on encoder for the patched self-attention layers
        self.encoder.spatial_matrix_temp = spatial_matrix
        try:
            if self.use_crf and not (self.is_bio and self.num_labels == 15):
                # Convert to integer buckets [0, 1000]
                x0_bucket = (x0 * 1000.0).round().long()
                y0_bucket = (y0 * 1000.0).round().long()
                x1_bucket = (x1 * 1000.0).round().long()
                y1_bucket = (y1 * 1000.0).round().long()
                
                # Retrieve coordinate embeddings
                layout_emb = (
                    self.x0_emb(x0_bucket)
                    + self.y0_emb(y0_bucket)
                    + self.x1_emb(x1_bucket)
                    + self.y1_emb(y1_bucket)
                )
                
                # Sum coordinate embeddings with text embeddings
                word_embeds = self.encoder.embeddings(input_ids)
                inputs_embeds = word_embeds + layout_emb
                
                # Run encoder with inputs_embeds
                enc = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            else:
                # Clean text representations for standard token-classification boundary detection
                enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            if hasattr(self.encoder, "spatial_matrix_temp"):
                delattr(self.encoder, "spatial_matrix_temp")
            
        h      = enc.last_hidden_state                    # (B, L, H)
        h      = self.spatial_fusion(h, spatial_features) # (B, L, H)
        h      = self.dropout(h)
        logits = self.classifier(h)                       # (B, L, num_labels)

        loss = None
        crf_loss = None
        decoded = None

        if labels is not None and not self.use_crf:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.num_labels), labels.view(-1)
            )
            
        if self.use_crf:
            crf_logits = logits if self.training else (logits * 2.0)
            if labels is not None:
                mask = (labels != -100)
                labels_crf = labels.clone()
                
                first_valid_indices = mask.float().argmax(dim=1)
                for batch_idx, start_idx in enumerate(first_valid_indices):
                    if start_idx > 0:
                        labels_crf[batch_idx, :start_idx] = 0
                        mask[batch_idx, :start_idx] = True # Force true so pytorch-crf initiates pathing
                
                labels_crf[~mask] = 0
                
                if mask.sum() == 0:
                    crf_loss = torch.tensor(0.0, requires_grad=True, device=logits.device)
                else:
                    crf_loss = -self.crf(crf_logits, labels_crf, mask=mask, reduction='token_mean')
                    if self.is_bio and self.num_labels == 15:
                        crf_loss = crf_loss * 0.005
                # Training decode uses the label-derived mask (CLS/SEP carry label=O=0 and
                # are already included with mask=True via the first_valid_indices patch above).
                crf_mask = mask
            else:
                # Inference path: CLS and SEP are forced to O via logit masking, and padding is excluded via attention_mask
                crf_logits = crf_logits.clone()
                special_mask = (input_ids == 0) | (input_ids == 2)
                crf_logits[special_mask, 1:] = -10000.0
                crf_logits[special_mask, 0] = 10.0
                crf_mask = attention_mask.bool()

            original_transitions = self.crf.transitions.data.clone()
            original_start = self.crf.start_transitions.data.clone()
            
            self.enforce_hard_constraints()
            decoded = self.crf.decode(crf_logits, mask=crf_mask)
            
            with torch.no_grad():
                self.crf.transitions.copy_(original_transitions)
                self.crf.start_transitions.copy_(original_start)

        if is_cpu_single_batch:
            if loss is not None:
                loss = loss.mean()
            logits = logits[:1]
            if decoded is not None:
                decoded = decoded[:1]

        return {
            "logits": logits,
            "loss": loss,
            "crf_loss": crf_loss,
            "decoded": decoded
        }


class SpatialBoundaryGate(nn.Module):
    """Deep Early-Fusion Multi-Class CRF emission trunk.

    Processes text hidden states and spatial layout features jointly via a
    1D convolutional sliding window, then projects the concatenated (768+128)
    fused representation directly to `num_labels` emission logits — replacing
    the former scalar B-SEG additive bias with a full sequence classifier.
    """
    def __init__(self, spatial_in_dim: int = 12, num_labels: int = 3, hidden_size: int = 384):
        super().__init__()
        # Sliding-window spatial encoder: captures layout transitions across
        # 5 adjacent tokens (kernel_size=5, same-padding)
        self.spatial_conv = nn.Conv1d(
            in_channels=spatial_in_dim, 
            out_channels=128, 
            kernel_size=5, # Expanded receptive field
            padding=2      # Maintain sequence length alignment (kernel 5 requires padding 2)
        )
        # Joint classification trunk: (hidden_size text + 128 spatial) → 3 classes
        # Refactor from nn.Linear(hidden_dim, 1) to support full 3-class logit distributions
        fused_feature_dim = hidden_size + 128
        self.classifier_head = nn.Linear(fused_feature_dim, 3)

    def forward(self, hidden_states: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        # spatial_features: (B, L, spatial_in_dim) → channels-first for Conv1d
        x = spatial_features.permute(0, 2, 1)          # (B, spatial_in_dim, L)
        x = torch.relu(self.spatial_conv(x))            # (B, 128, L)
        spatial_up_emb = x.permute(0, 2, 1)            # (B, L, 128)

        # Deep early-fusion: concatenate text + spatial embeddings
        fused = torch.cat([hidden_states, spatial_up_emb], dim=-1)  # (B, L, 896)
        return self.classifier_head(fused)                   # (B, L, 3)


class PhraseSegmenterTransformer(ResumeTokenClassifier):
    def __init__(self, num_labels: int, spatial_dim: int = SPATIAL_DIM, model_name: str = BACKBONE_NAME, dropout: float = DROPOUT, use_crf: bool = False):
        super().__init__(num_labels=num_labels, spatial_dim=spatial_dim, model_name=model_name, dropout=dropout, use_crf=use_crf, is_bio=True)
        self.num_labels = 3
        # Deep Early-Fusion gate is now the primary emission head for all classes
        self.boundary_gate = SpatialBoundaryGate(spatial_in_dim=spatial_dim, num_labels=self.num_labels, hidden_size=self.encoder.config.hidden_size)
        self._patch_segmenter_attention_bias()

    def _patch_segmenter_attention_bias(self):
        # We override the patched forward method of the attention layers
        for idx in [3, 4, 5]:
            if idx < len(self.encoder.encoder.layer):
                self_attn = self.encoder.encoder.layer[idx].attention.self
                # Bind our custom segmenter attention forward method
                self_attn.forward = types.MethodType(custom_segmenter_attention_forward, self_attn)

    def forward(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        spatial_features: torch.Tensor,
        labels:           torch.Tensor | None = None,
        spatial_matrix:   torch.Tensor | None = None,
    ):
        is_cpu_single_batch = (input_ids.shape[0] == 1 and input_ids.device.type == "cpu")
        if is_cpu_single_batch:
            input_ids = torch.cat([input_ids, input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
            spatial_features = torch.cat([spatial_features, spatial_features], dim=0)
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)
            if spatial_matrix is not None:
                spatial_matrix = torch.cat([spatial_matrix, spatial_matrix], dim=0)

        # Extract early layout coordinate features directly from the first 4 components of spatial_features
        x0 = torch.clamp(spatial_features[..., 0], 0.0, 1.0)
        y0 = torch.clamp(spatial_features[..., 1], 0.0, 1.0)
        x1 = torch.clamp(spatial_features[..., 2], 0.0, 1.0)
        y1 = torch.clamp(spatial_features[..., 3], 0.0, 1.0)

        # Store spatial_matrix on encoder for the patched self-attention layers
        self.encoder.spatial_matrix_temp = spatial_matrix
        try:
            # Convert to integer buckets [0, 1000]
            x0_bucket = (x0 * 1000.0).round().long()
            y0_bucket = (y0 * 1000.0).round().long()
            x1_bucket = (x1 * 1000.0).round().long()
            y1_bucket = (y1 * 1000.0).round().long()
            
            # Retrieve coordinate embeddings
            layout_emb = (
                self.x0_emb(x0_bucket)
                + self.y0_emb(y0_bucket)
                + self.x1_emb(x1_bucket)
                + self.y1_emb(y1_bucket)
            )
            
            # Sum coordinate embeddings with text embeddings (amplified spatial embeddings to contribute 35%)
            word_embeds = self.encoder.embeddings(input_ids)
            inputs_embeds = 0.65 * word_embeds + 0.35 * layout_emb
            
            # Run encoder with inputs_embeds
            enc = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        finally:
            if hasattr(self.encoder, "spatial_matrix_temp"):
                delattr(self.encoder, "spatial_matrix_temp")
            
        h = enc.last_hidden_state                    # (B, L, 768)
        h = self.spatial_fusion(h, spatial_features) # (B, L, 768)
        h = self.dropout(h)

        logits = self.boundary_gate(h, spatial_features)  # (B, L, 3)

        loss = None
        crf_loss = None
        decoded = None

        if labels is not None and not self.use_crf:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, 3), labels.view(-1)
            )

        if self.use_crf:
            crf_logits = logits if self.training else (logits * 2.0)
            if labels is not None:
                mask = (labels != -100)
                labels_crf = labels.clone()
                
                first_valid_indices = mask.float().argmax(dim=1)
                for batch_idx, start_idx in enumerate(first_valid_indices):
                    if start_idx > 0:
                        labels_crf[batch_idx, :start_idx] = 0
                        mask[batch_idx, :start_idx] = True
                
                labels_crf[~mask] = 0
                
                if mask.sum() == 0:
                    crf_loss = torch.tensor(0.0, requires_grad=True, device=logits.device)
                else:
                    crf_loss = -self.crf(crf_logits, labels_crf, mask=mask, reduction='token_mean')
                crf_mask = mask
            else:
                # Inference: exclude padding via attention_mask.
                # CLS/SEP retain their trained O-like emissions — no zeroing needed.
                crf_mask = attention_mask.bool()

            original_transitions = self.crf.transitions.data.clone()
            original_start = self.crf.start_transitions.data.clone()
            
            self.enforce_hard_constraints()
            decoded = self.crf.decode(crf_logits, mask=crf_mask)
            
            with torch.no_grad():
                self.crf.transitions.copy_(original_transitions)
                self.crf.start_transitions.copy_(original_start)

        if is_cpu_single_batch:
            if loss is not None:
                loss = loss.mean()
            if crf_loss is not None:
                crf_loss = crf_loss.mean()
            logits = logits[:1]

        if self.use_crf:
            res = {"logits": logits}
            if crf_loss is not None:
                res["crf_loss"] = crf_loss
            if decoded is not None:
                res["decoded"] = decoded
            return res
        else:
            if labels is not None:
                return loss, logits
            else:
                return logits
