from __future__ import annotations
import os
import re
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

SPATIAL_DIM = 20
_TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_TOKENIZER_WORD: AutoTokenizer | None = None
_TOKENIZER_SEQ:  AutoTokenizer | None = None

def _get_word_tokenizer() -> AutoTokenizer:
    global _TOKENIZER_WORD
    if _TOKENIZER_WORD is None:
        _TOKENIZER_WORD = AutoTokenizer.from_pretrained(_TOKENIZER_NAME, add_prefix_space=True)
    return _TOKENIZER_WORD

def _get_seq_tokenizer() -> AutoTokenizer:
    global _TOKENIZER_SEQ
    if _TOKENIZER_SEQ is None:
        _TOKENIZER_SEQ = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
    return _TOKENIZER_SEQ

def _normalize_spatial(tokens: list[dict], all_tokens: list[dict] | None = None, augment: bool = False) -> list[list[float]]:
    """11D spatial features per token + 3D relative derivatives: x0n y0n wn hn bold caps font_n abs_y wn dy_n heading_dist."""
    # Define core horizontal phrase breaks
    INLINE_DELIMITERS = {"-", "–", "—", "|", ",", "•", "●", "❖", "▪", ":", "~", "/", "(", ")"}
    
    current_line_idx = -1
    distance_from_delimiter = 0
    delimiter_dists = []
    
    for idx in range(len(tokens)):
        curr_tok = tokens[idx]
        tok_str = curr_tok.get("token", "").strip()
        line_idx = curr_tok.get("lineIndex", 0)
        
        if line_idx != current_line_idx:
            # Reset distance tracking on a brand new horizontal line canvas
            current_line_idx = line_idx
            distance_from_delimiter = 0
        elif tok_str in INLINE_DELIMITERS:
            # Reset counter when crossing an explicit layout anchor symbol
            distance_from_delimiter = 0
        else:
            distance_from_delimiter += 1
            
        # Normalize: cap at 10 and scale to [0.0, 1.0]
        norm_dist = min(float(distance_from_delimiter) / 10.0, 1.0)
        delimiter_dists.append(norm_dist)

    # Precompute relative layout derivatives
    deltas = []
    for idx in range(len(tokens)):
        if idx == 0:
            line_delta = 0.0
            font_delta = 0.0
            bold_delta = 0.0
        else:
            prev_tok = tokens[idx - 1]
            curr_tok = tokens[idx]
            
            line_delta = float(curr_tok.get("lineIndex", 0) - prev_tok.get("lineIndex", 0))
            font_delta = float(curr_tok.get("fontSize", 0.0) - prev_tok.get("fontSize", 0.0))
            bold_delta = float(int(curr_tok.get("isBold", False)) - int(prev_tok.get("isBold", False)))
            
            # Check for absolute page boundaries
            if curr_tok.get("page", 0) != prev_tok.get("page", 0):
                line_delta = 1.0 # Force clear break signal on page wraps
        deltas.append((line_delta, font_delta, bold_delta))

    ref_tokens = all_tokens if all_tokens is not None else tokens
    
    heading_lines = sorted(list({(t.get("page", 0), t.get("lineIndex", 0)) for t in ref_tokens if t.get("bioLabel") == "B-HEADING"}))
    page_max_lines = defaultdict(int)
    for t in ref_tokens:
        page_max_lines[t.get("page", 0)] = max(page_max_lines[t.get("page", 0)], t.get("lineIndex", 0))

    pages = defaultdict(list)
    for i, t in enumerate(tokens):
        pages[t.get("page", 0)].append((i, t))

    result = [None] * len(tokens)
    for items in pages.values():
        xs  = [t.get("x0", 0.0)       for _, t in items]
        x1s = [t.get("x1", 0.0)       for _, t in items]
        ys  = [t.get("y0", 0.0)       for _, t in items]
        y1s = [t.get("y1", 0.0)       for _, t in items]
        fs  = [t.get("fontSize", 9.0) for _, t in items]

        xmin, xmax = min(xs),  max(x1s)
        ymin, ymax = min(ys),  max(y1s)
        pw = max(xmax - xmin, 1e-6)
        ph = max(ymax - ymin, 1e-6)
        fmax = max(fs) or 1.0

        items_sorted = sorted(items, key=lambda x: x[0])
        prev_y1 = None
        prev_token = None
        for idx, t in items_sorted:
            x0n  = (t.get("x0", 0.0) - xmin) / pw
            y0n  = (t.get("y0", 0.0) - ymin) / ph
            wn   = (t.get("x1", 0.0) - t.get("x0", 0.0)) / pw
            hn   = (t.get("y1", 0.0) - t.get("y0", 0.0)) / ph
            
            # Calculate dx_n (horizontal leading margin) and is_line_start flag
            if prev_token is not None and t.get("page", 0) == prev_token.get("page", 0) and t.get("lineIndex", 0) == prev_token.get("lineIndex", 0):
                dx_n = t.get("x0", 0.0) - prev_token.get("x1", 0.0)
                is_line_start = 0.0
            else:
                dx_n = 0.0
                is_line_start = 1.0

            if augment:
                noise_x0 = random.gauss(0, 0.015)
                noise_x1 = random.gauss(0, 0.015)
                x0n_aug = max(min(x0n + noise_x0, 1.0), 0.0)
                x1n_aug = max(min((x0n + wn) + noise_x1, 1.0), 0.0)
                if x1n_aug < x0n_aug:
                    x1n_aug = x0n_aug
                x0n = x0n_aug
                wn = x1n_aug - x0n_aug
                
                noise_y0 = random.gauss(0, 0.005)
                noise_y1 = random.gauss(0, 0.005)
                y0n_aug = max(min(y0n + noise_y0, 1.0), 0.0)
                y1n_aug = max(min((y0n + hn) + noise_y1, 1.0), 0.0)
                if y1n_aug < y0n_aug:
                    y1n_aug = y0n_aug
                y0n = y0n_aug
                hn = y1n_aug - y0n_aug
                
            bold = float(t.get("isBold", False))
            text = t.get("token", "")
            caps = float(len(text) > 1 and text.isupper())
            fn   = t.get("fontSize", 9.0) / fmax
            abs_y = t.get("y0", 0.0) / max(max(y1s), 1.0)
            
            dy_n = 0.0 if prev_y1 is None else (t.get("y0", 0.0) - prev_y1) / ph

            is_after_inline_delimiter = 0.0
            inline_seps = {"|", "-", "—", "\\", "/", "(", ")"}
            if idx > 0 and tokens[idx - 1].get("token", "").strip() in inline_seps:
                is_after_inline_delimiter = 1.0
            elif idx > 1 and tokens[idx - 2].get("token", "").strip() in inline_seps:
                is_after_inline_delimiter = 1.0

            deg_pat = re.compile(r'\b(b\.?tech|m\.?c\.?a|b\.?c\.?a|m\.?b\.?a|bachelor|master|diploma|cbse|icse)\b', re.IGNORECASE)
            is_degree_anchor = 1.0 if deg_pat.search(text) else 0.0

            inst_pat = re.compile(r'\b(university|college|instit|school|academy)\b', re.IGNORECASE)
            is_institution_anchor = 1.0 if inst_pat.search(text) else 0.0
            
            connectives = {"and", "of", "in", "with", "from", "to"}
            is_connective_word = 1.0 if text.strip().lower() in connectives else 0.0
            
            result[idx] = [x0n, y0n, wn, hn, bold, caps, fn, abs_y, dx_n, dy_n, is_line_start, 0.0] + list(deltas[idx]) + [delimiter_dists[idx]] + [is_after_inline_delimiter, is_degree_anchor, is_institution_anchor, is_connective_word]
            prev_y1 = t.get("y1", 0.0)
            prev_token = t

    return [r if r is not None else [0.0] * SPATIAL_DIM for r in result]


def bio_to_bilou(bio_labels: list[str]) -> list[str]:
    bilou_labels = []
    n = len(bio_labels)
    for i, label in enumerate(bio_labels):
        if label == "O" or label == "NONE":
            bilou_labels.append(label)
        elif label.startswith("B-"):
            tag = label[2:]
            next_label = bio_labels[i+1] if i + 1 < n else "O"
            if next_label == f"I-{tag}":
                bilou_labels.append(f"B-{tag}")
            else:
                bilou_labels.append(f"U-{tag}")
        elif label.startswith("I-"):
            tag = label[2:]
            next_label = bio_labels[i+1] if i + 1 < n else "O"
            if next_label == f"I-{tag}":
                bilou_labels.append(f"I-{tag}")
            else:
                bilou_labels.append(f"L-{tag}")
        else:
            bilou_labels.append(label)
    return bilou_labels


def add_line_separators(
    tokens: list[dict],
    words: list[str],
    label_ids: list[int],
    spatial: list[list[float]],
    sep_token: str = "</s>"
) -> tuple[list[str], list[int], list[list[float]], list[int | None]]:
    new_words = []
    new_labels = []
    new_spatial = []
    original_map = []
    
    for idx, token in enumerate(tokens):
        if token.get("tokenIndex", 0) == 0 and idx > 0:
            new_words.append(sep_token)
            new_labels.append(-100)
            new_spatial.append([0.0] * SPATIAL_DIM)
            original_map.append(None)
            
        new_words.append(words[idx])
        new_labels.append(label_ids[idx])
        new_spatial.append(spatial[idx])
        original_map.append(idx)
        
    return new_words, new_labels, new_spatial, original_map


PROJECT_LABEL_MAP = {
    "B-PROJ_NAME": "PROJ", "I-PROJ_NAME": "PROJ",
    "B-PROJ_COMPANY": "COMP", "I-PROJ_COMPANY": "COMP",
    "B-SDATE": "SDATE", "I-SDATE": "SDATE",
    "B-EDATE": "EDATE", "I-EDATE": "EDATE",
    "B-DESC": "DESC", "I-DESC": "DESC"
}

def translate_project_label(bio: str) -> str:
    if bio in ("B-URL", "I-URL", "B-PROJ_TECH", "I-PROJ_TECH", "B-LINK", "I-LINK", "URL", "PROJ_TECH", "LINK"):
        return "O"
    if bio in PROJECT_LABEL_MAP:
        prefix = "B-" if bio.startswith("B-") else "I-"
        return prefix + PROJECT_LABEL_MAP[bio]
    return bio


EDUCATION_LABEL_MAP = {
    "B-DEG": "DEG", "I-DEG": "DEG",
    "B-DEGREE": "DEG", "I-DEGREE": "DEG",
    "B-INST": "INST", "I-INST": "INST",
    "B-INSTITUTION": "INST", "I-INSTITUTION": "INST",
    "B-SDATE": "SDATE", "I-SDATE": "SDATE",
    "B-EDATE": "EDATE", "I-EDATE": "EDATE",
    "B-DESC": "DESC", "I-DESC": "DESC",
    "B-EDESC": "DESC", "I-EDESC": "DESC",
    "B-GPA": "GPA", "I-GPA": "GPA",
    "B-GRADE": "GPA", "I-GRADE": "GPA",
    "B-SCORE": "GPA", "I-SCORE": "GPA"
}

def translate_education_label(bio: str) -> str:
    label = bio.strip().upper() if bio else "O"
    if label in EDUCATION_LABEL_MAP:
        prefix = "B-" if label.startswith("B-") else "I-"
        return prefix + EDUCATION_LABEL_MAP[label]
    return label




class _TokenClassificationDataset(Dataset):
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer  = _get_word_tokenizer()
        self.items: list[dict] = []
        self.lab2id: dict[str, int] = {}
        self.id2lab: dict[int, str] = {}
        self.spatial_dim = 16

    def __len__(self) -> int:
        return len(self.items)

    def _encode(self, words: list[str], label_ids: list[int], spatial: list[list[float]]) -> dict:
        words_clean = [w.replace('"', '').replace("'", "").replace("‘", "").replace("’", "") for w in words]
        enc = self.tokenizer(
            words_clean,
            is_split_into_words=True,
            return_offsets_mapping=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        word_ids       = enc.word_ids(0)
        seq_len        = input_ids.shape[0]

        labels_out = []
        prev_wid = None
        
        id2lab = getattr(self, "id2lab", {})
        lab2id = getattr(self, "lab2id", {})

        # Structural layout tokens that MUST propagate gradients during training.
        # These are geometric anchors (bullets, delimiters, separators) that mark
        # sequence boundaries — masking them prevents the network from ever learning
        # to predict transitions at these positions.
        STRUCTURAL_TOKENS = {
            "|", "•", "-", "–", "—", "*", "▪", "◦", "■", "·", "✓", "✔",
            "\uf0b7", "\uf0a7", ","
        }

        for wid in word_ids:
            if wid is None:
                labels_out.append(-100)
            else:
                token_str = words[wid] if wid < len(words) else ""

                # Check if token carries alphanumeric content or is a structural delimiter
                is_alphanumeric = bool(re.search(r'[a-zA-Z0-9]', token_str))
                is_phase3 = self.__class__.__name__ in ("Phase3SegmentDataset", "Phase3ProjectSegmentDataset")
                is_structural = is_phase3 and (
                    token_str in STRUCTURAL_TOKENS
                    or (len(token_str) == 1 and token_str in STRUCTURAL_TOKENS)
                )

                # Mask out pure noise punctuation; exempt structural anchors for Phase 3 only
                if not is_alphanumeric and not is_structural:
                    if is_phase3:
                        labels_out.append(0)
                    else:
                        labels_out.append(-100)
                elif wid != prev_wid:
                    word_label = label_ids[wid] if wid < len(label_ids) else 0
                    labels_out.append(word_label)
                else:
                    # Mask out subsequent subwords with -100 to align with word-level evaluation
                    labels_out.append(-100)

            prev_wid = wid

        spatial_t = torch.zeros(seq_len, self.spatial_dim, dtype=torch.float32)
        for i, wid in enumerate(word_ids):
            if wid is not None and wid < len(spatial):
                spatial_t[i] = torch.tensor(spatial[wid][:self.spatial_dim], dtype=torch.float32)

        x0 = torch.tensor([spatial[wid][0] if (wid is not None and wid < len(spatial)) else 0.0 for wid in word_ids], dtype=torch.float32)
        y0 = torch.tensor([spatial[wid][1] if (wid is not None and wid < len(spatial)) else 0.0 for wid in word_ids], dtype=torch.float32)
        dx = x0.unsqueeze(1) - x0.unsqueeze(0)
        dy = y0.unsqueeze(1) - y0.unsqueeze(0)
        spatial_matrix = torch.stack([dx, dy], dim=-1)

        return {
            "input_ids":        input_ids,
            "attention_mask":   attention_mask,
            "labels":           torch.tensor(labels_out, dtype=torch.long),
            "spatial_features": spatial_t,
            "spatial_matrix":   spatial_matrix,
        }

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        base = self._encode(item["words"], item["label_ids"], item["spatial"])
        base["doc_id"] = item["doc_id"]
        return base


def collate_fn(batch: list[dict]) -> dict:
    res = {
        "input_ids":        torch.stack([b["input_ids"]        for b in batch]),
        "attention_mask":   torch.stack([b["attention_mask"]   for b in batch]),
        "labels":           torch.stack([b["labels"]           for b in batch]),
        "spatial_features": torch.stack([b["spatial_features"] for b in batch]),
        "doc_ids":          [b["doc_id"]                       for b in batch],
    }
    if "spatial_matrix" in batch[0]:
        res["spatial_matrix"] = torch.stack([b["spatial_matrix"] for b in batch])
    return res


def collate_chunk_fn(batch: list[dict]) -> dict:
    return {
        "input_ids":        torch.stack([b["input_ids"]        for b in batch]),
        "attention_mask":   torch.stack([b["attention_mask"]   for b in batch]),
        "labels":           torch.stack([b["label"]            for b in batch]),
        "spatial_features": torch.stack([b["spatial_features"] for b in batch]),
        "doc_ids":          [b["doc_id"]                       for b in batch],
        "prev_labels":      torch.stack([b["prev_label"]       for b in batch]),
    }
