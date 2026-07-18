"""Phase 4 Chunk Classification Head — ResumeChunkClassifier.

Architecture
------------
Text path
  DistilRoBERTa backbone (frozen or fine-tuned at lower LR)
  → Mean-pool hidden states across non-padding positions
  → 768-D phrase representation

Spatial path
  12-D averaged spatial layout vector (x0n, y0n, wn, hn + 8 auxiliary dims)
  → Linear 12 → 128 → GELU → Dropout(0.1)  [128-D spatial projection]

Fusion & Classification
  [text_emb(768) ‖ spatial_proj(128)] → 896-D concat
  → GLU-style sigmoid gate → 896-D fused representation
  → Linear 896 → 5-class logits

Target label map — canonical 5-class schema, no ambient background padding:
    {"DESC": 0, "ROLE": 1, "COMP": 2, "SDATE": 3, "EDATE": 4}

Usage
-----
    model = ResumeChunkClassifier(num_labels=5)
    out   = model(input_ids, attention_mask, spatial_features, labels=y)
    loss, logits = out["loss"], out["logits"]
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

BACKBONE_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
DROPOUT         = 0.1
SPATIAL_DIM     = 12
PROJ_DIM        = 128   # spatial projection width
PREV_LABEL_DIM  = 32    # prev_label context embedding dimension

# ── Phase 4 canonical 5-class label schema (no background padding) ──────────
# Ordering locked to: {"DESC": 0, "ROLE": 1, "COMP": 2, "SDATE": 3, "EDATE": 4}
CHUNK4_LABELS: list[str] = ["DESC", "ROLE", "COMP", "SDATE", "EDATE"]
CHUNK4_2ID:    dict[str, int] = {"DESC": 0, "ROLE": 1, "COMP": 2, "SDATE": 3, "EDATE": 4}
ID2_CHUNK4:    dict[int, str] = {0: "DESC", 1: "ROLE", 2: "COMP", 3: "SDATE", 4: "EDATE"}
NUM_CHUNK4: int = 5


class _SpatialLayoutProjector(nn.Module):
    """128-D projected spatial layout block.

    Projects a ``(B, spatial_dim)`` coordinate vector through a two-layer
    MLP with GELU activation and dropout into a 128-dimensional embedding
    that can be fused with text representations.
    """

    def __init__(self, spatial_dim: int = SPATIAL_DIM, proj_dim: int = PROJ_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(spatial_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spatial_features : FloatTensor  (B, spatial_dim) or (B, L, spatial_dim)
            Phrase-level averaged spatial coordinates.  If a 3-D tensor is
            received (per-token spatial from the token-level pipeline), the
            first token's row is taken as the block anchor.

        Returns
        -------
        FloatTensor  (B, proj_dim)
        """
        if spatial_features.dim() == 3:
            # Block-level anchor: use first token's spatial row
            spatial_features = spatial_features[:, 0, :]
        return self.mlp(spatial_features)   # (B, proj_dim)


class _MultiHeadClassificationLayer(nn.Module):
    """Final 5-class multi-head classification layer.

    Receives the 896-D (768 text + 128 spatial) fused representation
    and projects it to ``num_labels`` logits through a gated pathway that
    models cross-modal interactions.
    """

    def __init__(self, fused_dim: int, num_labels: int, dropout: float = DROPOUT):
        super().__init__()
        mid = fused_dim // 2
        self.gate   = nn.Linear(fused_dim, fused_dim)
        self.hidden = nn.Sequential(
            nn.Linear(fused_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(mid, num_labels)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        fused : FloatTensor  (B, fused_dim)

        Returns
        -------
        logits : FloatTensor  (B, num_labels)
        """
        gate   = torch.sigmoid(self.gate(fused))
        gated  = gate * fused
        hidden = self.hidden(gated)
        return self.head(hidden)


class ResumeChunkClassifier(nn.Module):
    """Phase 4: Chunk-level experience entity classifier.

    Architecture
    ------------
    Text path
      DistilRoBERTa backbone → mean-pool → 768-D phrase representation

    Spatial path
      12-D averaged layout vector → MLP → 128-D spatial projection

    Prev-label context path
      Previous block's class index → nn.Embedding → 32-D context embedding
      Provides sequential ordering signal to disambiguate SDATE vs EDATE,
      which share identical surface text but differ only in position within
      an entry sequence.

    Fusion
      [text(768) ‖ spatial(128) ‖ prev_label(32)] → 928-D
      → GLU-gated hidden → 5-class logits

    Parameters
    ----------
    num_labels :
        Number of output classes (default 5 → DESC/ROLE/COMP/SDATE/EDATE).
    spatial_dim :
        Input dimension of spatial feature vector (default 12).
    model_name :
        HuggingFace model identifier for the encoder backbone.
    dropout :
        Dropout probability.
    """

    LABEL_NAMES = CHUNK4_LABELS   # ["DESC", "ROLE", "COMP", "SDATE", "EDATE"]

    def __init__(
        self,
        num_labels:     int   = NUM_CHUNK4,
        spatial_dim:    int   = SPATIAL_DIM,
        model_name:     str   = BACKBONE_NAME,
        dropout:        float = DROPOUT,
        prev_label_dim: int   = PREV_LABEL_DIM,
    ):
        super().__init__()
        self.num_labels     = num_labels
        self.spatial_dim    = spatial_dim
        self.prev_label_dim = prev_label_dim

        # ── Text encoder ─────────────────────────────────────────────────────
        self.encoder = AutoModel.from_config(
            AutoConfig.from_pretrained(model_name, local_files_only=True)
        )
        h = self.encoder.config.hidden_size          # 768 for distilroberta-base

        # ── 128-D Spatial Layout Block ────────────────────────────────────────
        self.spatial_proj = _SpatialLayoutProjector(
            spatial_dim=spatial_dim,
            proj_dim=PROJ_DIM,
            dropout=dropout,
        )

        # ── 32-D Prev-label Context Embedding ────────────────────────────────
        # num_labels + 1 rows: indices 0‥num_labels-1 are real labels;
        # index num_labels is the warm-start / unknown sentinel.
        self.prev_label_emb = nn.Embedding(
            num_embeddings=num_labels + 1,
            embedding_dim=prev_label_dim,
            padding_idx=None,
        )
        # Heavy dropout on the autoregressive context path — p=0.40 breaks
        # state-locking feedback loops where prev_label cascades into false
        # COMP/ROLE classifications on tech-keyword description blocks.
        self.context_dropout = nn.Dropout(p=0.40)

        self.dropout = nn.Dropout(dropout)

        # ── 5-class Multi-Head Classification Layer ───────────────────────────
        # fused_dim = text(768) + spatial(128) + prev_label(32) = 928
        fused_dim = h + PROJ_DIM + prev_label_dim
        self.classifier = _MultiHeadClassificationLayer(
            fused_dim=fused_dim,
            num_labels=num_labels,
            dropout=dropout,
        )

    # ── Mean-pool helper ─────────────────────────────────────────────────────

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask:    torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention-masked mean over the sequence dimension.

        Parameters
        ----------
        last_hidden_state : (B, L, H)
        attention_mask    : (B, L)

        Returns
        -------
        FloatTensor (B, H)
        """
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        ).float()
        sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask   = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_hidden / sum_mask

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        spatial_features: torch.Tensor,
        labels:           torch.Tensor | None = None,
        prev_labels:      torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """
        Parameters
        ----------
        input_ids :        LongTensor    (B, L)
        attention_mask :   LongTensor    (B, L)
        spatial_features : FloatTensor   (B, 12)
        labels :           LongTensor    (B,) — class indices 0-4, optional
        prev_labels :      LongTensor    (B,) — previous block's class index;
                           use ``num_labels`` (=5) as the warm-start sentinel
                           for the first block in an entry.

        Returns
        -------
        dict with keys:
            "logits"  : FloatTensor (B, num_labels)
            "loss"    : FloatTensor scalar | None
        """
        B = input_ids.shape[0]
        dev = input_ids.device

        # ── Build prev_label index (clamp sentinel to num_labels) ─────────────
        if prev_labels is None:
            # Default warm-start: index = num_labels (the sentinel row)
            prev_idx = torch.full((B,), self.num_labels, dtype=torch.long, device=dev)
        else:
            prev_idx = prev_labels.to(dev).clamp(0, self.num_labels)

        # ── CPU single-batch SIGBUS workaround (mirrors existing models) ──────
        is_cpu_single = (B == 1 and dev.type == "cpu")
        if is_cpu_single:
            input_ids        = torch.cat([input_ids,        input_ids],        dim=0)
            attention_mask   = torch.cat([attention_mask,   attention_mask],   dim=0)
            spatial_features = torch.cat([spatial_features, spatial_features], dim=0)
            prev_idx         = torch.cat([prev_idx,         prev_idx],         dim=0)
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)

        # ── Text encoding ─────────────────────────────────────────────────────
        enc      = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = self._mean_pool(enc.last_hidden_state, attention_mask)  # (B, 768)
        text_emb = self.dropout(text_emb)

        # ── Spatial projection ────────────────────────────────────────────────
        spatial_emb = self.spatial_proj(spatial_features)   # (B, 128)
        spatial_emb = self.dropout(spatial_emb)

        # ── Prev-label context embedding (heavy context_dropout=0.40) ────────────
        ctx_emb = self.prev_label_emb(prev_idx)             # (B, 32)
        ctx_emb = self.context_dropout(ctx_emb)             # p=0.40 regularization

        # ── Fusion & classification ───────────────────────────────────────────
        # [text(768) ‖ spatial(128) ‖ prev_label(32)] = 928-D
        fused  = torch.cat([text_emb, spatial_emb, ctx_emb], dim=-1)
        logits = self.classifier(fused)                        # (B, num_labels)

        # ── Loss ──────────────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        # ── Undo SIGBUS duplication ───────────────────────────────────────────
        if is_cpu_single:
            logits = logits[:1]
            if loss is not None:
                loss = loss.mean()

        return {"logits": logits, "loss": loss}
