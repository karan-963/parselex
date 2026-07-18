"""Segment-based skills token inference (training evaluate.py parity)."""

from __future__ import annotations

import re

import torch

from .config import ID2LABEL, MAX_SEG_LEN, MAX_SEGS, SPATIAL_DIM
from ..confidence import word_level_confidences


_MAJOR_STRUCTURAL_HEADERS = {
    "experience", "work experience", "professional experience", "employment history",
    "education", "academic background", "academics",
    "projects", "academic projects", "key projects", "personal projects",
    "summary", "professional summary", "career objective", "objective", "profile",
    "personal details", "personal info", "personal information", "interests",
    "certifications", "hobbies", "declarations", "declaration", "languages",
}

_SKILLS_HEADER_PATTERNS = {
    "skills", "technical skills", "skill summary", "technologies",
    "technology", "core competencies", "technical_skills", "competencies",
}


def _expand_skills_segments(segments: list[dict], helpers: dict) -> list[tuple[dict, bool]]:
    in_skills = False
    expanded: list[tuple[dict, bool]] = []
    for segment in segments:
        text = (segment.get("text") or "").strip().lower()
        text_clean = re.sub(r"[^a-z\s]", "", text).strip()
        if text_clean in _SKILLS_HEADER_PATTERNS or helpers["is_skills_segment"](segment):
            in_skills = True
        if in_skills and text_clean in _MAJOR_STRUCTURAL_HEADERS and ":" not in text:
            in_skills = False
        if in_skills:
            for sub_seg in helpers["split_skills_segment"](segment):
                expanded.append((sub_seg, True))
        else:
            expanded.append((segment, False))
    return expanded


@torch.no_grad()
def infer_token_labels(
    resume_tokens: list[dict],
    model: torch.nn.Module,
    tokenizer,
    helpers: dict,
    device: torch.device,
    all_tokens: list[dict] | None = None,
) -> tuple[list[str], dict[int, float]]:
    """Run SkillsSegmentClassifierModel and map subword preds back to resume tokens."""
    empty = (["O"] * len(resume_tokens), {})
    cleaned = helpers["clean_cid_tokens"](resume_tokens)
    segments = helpers["construct_sentences_by_appearance"](cleaned)
    if not segments:
        return empty

    expanded_segments = _expand_skills_segments(segments, helpers)
    filtered_segments: list[dict] = []
    seg_skills_map: dict[int, bool] = {}
    for segment, is_skills in expanded_segments:
        if helpers["is_heading_segment"](segment):
            continue
        filtered_segments.append(segment)
        seg_skills_map[id(segment)] = is_skills

    if not filtered_segments:
        return empty

    spatial_source = all_tokens if all_tokens is not None else resume_tokens
    seg_input_ids: list[torch.Tensor] = []
    seg_attn_mask: list[torch.Tensor] = []
    spatial_list: list[list[float]] = []

    for segment in filtered_segments[:MAX_SEGS]:
        enc = tokenizer(
            segment["text"],
            max_length=MAX_SEG_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        seg_input_ids.append(enc["input_ids"].squeeze(0))
        seg_attn_mask.append(enc["attention_mask"].squeeze(0))
        spatial_list.append(
            helpers["extract_segment_spatial"](segment, all_tokens=spatial_source, spatial_dim=SPATIAL_DIM)
        )

    num_segs = len(seg_input_ids)
    if num_segs < MAX_SEGS:
        pad_tok = torch.full((MAX_SEG_LEN,), tokenizer.pad_token_id, dtype=torch.long)
        pad_mask = torch.zeros(MAX_SEG_LEN, dtype=torch.long)
        for _ in range(MAX_SEGS - num_segs):
            seg_input_ids.append(pad_tok.clone())
            seg_attn_mask.append(pad_mask.clone())
            spatial_list.append([0.0] * SPATIAL_DIM)

    input_ids_t = torch.stack(seg_input_ids).unsqueeze(0).to(device)
    attn_mask_t = torch.stack(seg_attn_mask).unsqueeze(0).to(device)
    spatial_t = torch.tensor(spatial_list, dtype=torch.float32).unsqueeze(0).to(device)
    logits = model(input_ids_t, attn_mask_t, spatial_t)
    seg_logits = logits.squeeze(0)
    pred_ids = seg_logits.argmax(-1).cpu().tolist()

    orig_tok_id_to_pred: dict[int, str] = {}
    orig_tok_id_to_conf: dict[int, float] = {}
    for seg_idx, segment in enumerate(filtered_segments[:MAX_SEGS]):
        is_skills = seg_skills_map.get(id(segment), False)
        seg_tok_list = segment.get("tokens", [])
        if not is_skills:
            for token in seg_tok_list:
                orig_tok_id_to_pred[id(token)] = "O"
                orig_tok_id_to_conf[id(token)] = 0.0
            continue

        enc = tokenizer(
            segment["text"],
            max_length=MAX_SEG_LEN,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        word_ids = enc.word_ids()
        seg_pred_ids = pred_ids[seg_idx]
        word_pred: dict[int, str] = {}
        word_confs = word_level_confidences(seg_logits[seg_idx], word_ids, len(seg_tok_list))
        seen_wids: set[int] = set()
        for slot_i, wid in enumerate(word_ids):
            if wid is None or wid in seen_wids:
                continue
            seen_wids.add(wid)
            word_pred[wid] = ID2LABEL.get(seg_pred_ids[slot_i], "O")

        for word_pos, token in enumerate(seg_tok_list):
            orig_tok_id_to_pred[id(token)] = word_pred.get(word_pos, "O")
            orig_tok_id_to_conf[id(token)] = word_confs[word_pos] if word_pos < len(word_confs) else 0.0

    result = [orig_tok_id_to_pred.get(id(token), "O") for token in resume_tokens]
    return result, orig_tok_id_to_conf
