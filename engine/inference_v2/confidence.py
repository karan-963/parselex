"""Generic softmax confidence helpers for inference model predictions."""

from __future__ import annotations

import torch


def round_conf(value: float) -> float:
    return round(float(value), 4)


def max_prob(logits: torch.Tensor, pred_id: int) -> float:
    """Softmax probability of a single predicted class from a 1-D logit vector."""
    if logits.dim() > 1:
        logits = logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    idx = int(pred_id)
    if idx < 0 or idx >= probs.shape[-1]:
        return 0.0
    return round_conf(float(probs[idx].item()))


def batch_max_probs(logits: torch.Tensor, pred_ids: list[int]) -> list[float]:
    """Per-row max prob for a batch of predictions (B, num_classes) or (B,)."""
    if logits.dim() == 1:
        return [max_prob(logits, pred_ids[0])] if pred_ids else []
    probs = torch.softmax(logits, dim=-1)
    out: list[float] = []
    for row, pid in enumerate(pred_ids):
        if row >= probs.shape[0]:
            out.append(0.0)
            continue
        idx = int(pid)
        if idx < 0 or idx >= probs.shape[-1]:
            out.append(0.0)
        else:
            out.append(round_conf(float(probs[row, idx].item())))
    return out


def word_level_confidences(
    logits: torch.Tensor,
    word_ids: list[int | None],
    n_words: int,
    *,
    seq_dim: int = 0,
) -> list[float]:
    """Map subword logits to per-word confidence (first subword per word wins).

    ``logits`` shape (seq_len, num_classes) when seq_dim=0, or (1, seq_len, C) when batched.
    """
    if logits.dim() == 3:
        logits = logits[0]
    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    word_conf: dict[int, float] = {}
    for i, wid in enumerate(word_ids):
        if wid is None or wid >= n_words or wid in word_conf:
            continue
        pid = int(preds[i].item())
        word_conf[wid] = round_conf(float(probs[i, pid].item()))
    return [word_conf.get(wid, 0.0) for wid in range(n_words)]
