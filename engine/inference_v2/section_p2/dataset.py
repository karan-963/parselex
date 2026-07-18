import os
import re
import json
import random
import sys
from collections import defaultdict, Counter
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from . import config

# _PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "phase1"))
# if _PHASE1_DIR not in sys.path:
#     sys.path.append(_PHASE1_DIR)
# from excluded_resumes import is_excluded, list_active_json_files, load_excluded_ids

SECTION_CHUNK_LABELS = ["PERSONAL", "SUMMARY", "EXPERIENCE", "EDUCATION", "PROJECT", "SKILLS", "OTHER"]
CHUNK2ID             = {l: i for i, l in enumerate(SECTION_CHUNK_LABELS)}
ID2CHUNK             = {i: l for i, l in enumerate(SECTION_CHUNK_LABELS)}
NUM_CHUNK            = len(SECTION_CHUNK_LABELS)

HEADING_DICT_SINGLE = {
    "summary", "objective", "profile", "experience", "education", "skills",
    "projects", "certifications", "certification", "achievements", "awards",
    "publications", "research", "volunteering", "interests", "hobbies",
    "languages", "references", "contact", "competencies", "expertise",
    "employment", "qualifications",
}

_SKILLS_ROOTS          = ["skill", "tool", "technolog", "language", "competenc", "expertis"]
_OTHER_ROOTS_GUARDED   = ["certif", "award", "achiev", "honor", "activit"]   # blocked if EDUCATION
_OTHER_ROOTS_UNGUARDED = ["interest", "hobbie"]                               # always override
_EXPERIENCE_ROOTS      = ["intern", "employ", "work histor"]
_SUMMARY_ROOTS         = ["summar", "objective", "profil", "about me", "career statement"]

def _normalize_chunk_label(heading_text: str, current_label: str) -> str:
    h = heading_text.lower().strip()
    if any(root in h for root in _SKILLS_ROOTS):
        return "SKILLS"
    if any(root in h for root in _OTHER_ROOTS_UNGUARDED):
        return "OTHER"
    if current_label not in ("EDUCATION",) and any(root in h for root in _OTHER_ROOTS_GUARDED):
        return "OTHER"
    if current_label == "OTHER" and any(root in h for root in _EXPERIENCE_ROOTS):
        return "EXPERIENCE"
    if current_label == "PERSONAL" and any(root in h for root in _SUMMARY_ROOTS):
        return "SUMMARY"
    return current_label

def _merge_multiword_headings(tokens: list[dict]) -> list[dict]:
    result = [dict(t) for t in tokens]
    i = 0
    while i < len(result):
        if result[i].get("bioLabel") == "B-HEADING":
            j = i + 1
            h_page = result[i].get("page")
            h_line = result[i].get("lineIndex")
            while j < len(result):
                nxt = result[j]
                if (
                    nxt.get("page") == h_page
                    and nxt.get("lineIndex") == h_line
                    and nxt.get("bioLabel", "O") == "O"
                    and nxt.get("isBold", False)
                    and nxt["token"].strip().isupper()
                    and nxt["token"].strip().isalpha()
                ):
                    result[j] = dict(nxt)
                    result[j]["bioLabel"] = "I-HEADING"
                    j += 1
                else:
                    break
            i = j
        else:
            i += 1
    return result

def _aggregate_chunks(tokens: list[dict]) -> list[dict]:
    tokens = _merge_multiword_headings(tokens)
    line_first_idx = {}
    for t in tokens:
        key = (t.get("page"), t.get("lineIndex"))
        tidx = t.get("tokenIndex", 0)
        if key not in line_first_idx or tidx < line_first_idx[key]:
            line_first_idx[key] = tidx

    chunks = []
    current = None
    pre_heading = []

    for t in tokens:
        bio = t.get("bioLabel", "O")

        if (
            bio == "O"
            and current is not None
            and t.get("isBold", False)
            and t["token"].lower().strip() in HEADING_DICT_SINGLE
            and t.get("tokenIndex", 0) == line_first_idx.get((t.get("page"), t.get("lineIndex")), -1)
        ):
            bio = "B-HEADING"

        if bio in ("B-HEADING", "I-HEADING") and bio == "B-HEADING":
            if current is None and pre_heading:
                sections = [
                    tok.get("section") for tok in pre_heading
                    if tok.get("section") and tok.get("section") != "NONE"
                ]
                if sections:
                    majority = Counter(sections).most_common(1)[0][0]
                    if majority == "PERSONAL":
                        chunks.append({"heading": "[SECTION_START]", "tokens": pre_heading, "virtual": True})
                else:
                    chunks.append({"heading": "[SECTION_START]", "tokens": pre_heading, "virtual": True})
            if current is not None:
                chunks.append(current)
            current = {"heading": t["token"], "tokens": [t]}
        elif bio == "I-HEADING" and current is not None:
            current["heading"] = current["heading"] + " " + t["token"]
            current["tokens"].append(t)
        elif current is None:
            pre_heading.append(t)
        else:
            current["tokens"].append(t)

    if current is not None:
        chunks.append(current)
    return chunks

def sort_tokens_by_reading_order(tokens: list[dict]) -> list[dict]:
    return sorted(tokens, key=lambda t: (t.get("page", 0), t.get("y0", 0.0), t.get("x0", 0.0)))

def median_inter_token_gap(line_map: dict) -> float:
    gaps = []
    for _, line_tokens in line_map.items():
        if len(line_tokens) < 2:
            continue
        line_tokens_sorted = sorted(line_tokens, key=lambda x: x.get("x0", 0.0))
        for idx in range(1, len(line_tokens_sorted)):
            prev = line_tokens_sorted[idx - 1]
            curr = line_tokens_sorted[idx]
            gap = curr.get("x0", 0.0) - prev.get("x1", 0.0)
            if gap > 0:
                gaps.append(gap)
    return float(torch.median(torch.tensor(gaps)).item()) if gaps else 4.0

def median_line_height_of(tokens: list[dict]) -> float:
    heights = [t.get("y1", 0.0) - t.get("y0", 0.0) for t in tokens if t.get("y1", 0.0) > t.get("y0", 0.0)]
    return float(torch.median(torch.tensor(heights)).item()) if heights else 10.0

def generate_spatial_features(
    tokens: list[dict], 
    doc_median_gap: float, 
    doc_median_line_height: float, 
    augment: bool = False
) -> list[list[float]]:
    if not tokens:
        return []
    xs = [t.get("x0", 0.0) for t in tokens]
    x1s = [t.get("x1", 0.0) for t in tokens]
    ys = [t.get("y0", 0.0) for t in tokens]
    y1s = [t.get("y1", 0.0) for t in tokens]
    fs = [t.get("fontSize", 9.0) for t in tokens]
    xmin, xmax = min(xs), max(x1s)
    ymin, ymax = min(ys), max(y1s)
    pw = max(xmax - xmin, 1e-6)
    ph = max(ymax - ymin, 1e-6)
    fmax = max(fs) or 1.0

    BULLETS = {"•", "*", "-", "▪", "◦", "■", "–", "—", "\uf0b7", "·", "✓", "✔", "\uf0a7"}
    line_map = defaultdict(list)
    page_max_lines = defaultdict(int)
    for t in tokens:
        key = (t.get("page", 0), t.get("lineIndex", 0))
        line_map[key].append(t)
        page_max_lines[t.get("page", 0)] = max(page_max_lines[t.get("page", 0)], t.get("lineIndex", 0))
        
    line_counts = {k: len(v) for k, v in line_map.items()}
    line_upper = {}
    line_has_bullet = {}
    for k, lts in line_map.items():
        text = "".join(t.get("token", "") for t in lts)
        line_upper[k] = float(len(text) > 1 and text.isupper())
        lts_sorted = sorted(lts, key=lambda x: x.get("x0", 0.0))
        first_tok = lts_sorted[0].get("token", "") if lts_sorted else ""
        line_has_bullet[k] = float(first_tok in BULLETS)

    spatial_feats = []
    
    for idx, t in enumerate(tokens):
        x0_val = t.get("x0", 0.0) + (random.gauss(0, 1.0) if augment else 0.0)
        y0_val = t.get("y0", 0.0) + (random.gauss(0, 0.5) if augment else 0.0)
        
        dx = 0.0
        dy = 0.0
        is_new_line = 1.0
        
        if idx > 0:
            prev_t = tokens[idx - 1]
            if t.get("page") != prev_t.get("page"):
                dx = 0.0
                dy = y0_val / ph
                is_new_line = 1.0
            else:
                dx = (x0_val - prev_t.get("x1", 0.0)) / doc_median_gap
                dy = (y0_val - prev_t.get("y1", 0.0)) / doc_median_line_height
                is_new_line = float(t.get("lineIndex", 0) != prev_t.get("lineIndex", 0))

        bold = float(t.get("isBold", False))
        text = str(t.get("token", ""))
        caps = float(len(text) > 1 and text.isupper())
        fn = t.get("fontSize", 9.0) / fmax

        key = (t.get("page", 0), t.get("lineIndex", 0))
        line_len_feat = min(line_counts[key], 15) / 15.0
        is_line_upper = line_upper[key]
        is_bullet_start = line_has_bullet[key]
        
        page = t.get("page", 0)
        max_li = page_max_lines[page]
        line_rank_norm = t.get("lineIndex", 0) / max(max_li, 1.0)

        feat = [
            dx, dy, is_new_line, bold, caps, fn,
            line_len_feat, is_line_upper, is_bullet_start, line_rank_norm
        ]
        spatial_feats.append(feat)
        
    return spatial_feats

class SectionChunkDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", max_length: int = 256, augment: bool = False):
        self.max_length = max_length
        self.items = []
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.doc_meta = {}

        print(f"[DATA] Loading dataset split '{split}' from {data_dir}...")
        if not os.path.exists(data_dir):
            print(f"[DATA] Directory {data_dir} does not exist!")
            return
            
        excluded = load_excluded_ids()
        files = list_active_json_files(data_dir, excluded)

        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            resume_id = data.get("resumeId", os.path.splitext(os.path.basename(fpath))[0])
            if is_excluded(resume_id, excluded):
                continue

            raw_tokens = data.get("tokens", [])
            tokens = sort_tokens_by_reading_order(raw_tokens)
            if len(tokens) < 5:
                continue
                
            line_map = defaultdict(list)
            for t in tokens:
                line_map[(t.get("page", 0), t.get("lineIndex", 0))].append(t)
                
            doc_median_gap = median_inter_token_gap(line_map)
            doc_median_line_height = median_line_height_of(tokens)
            
            spatial = generate_spatial_features(
                tokens, doc_median_gap, doc_median_line_height, augment=augment
            )
            
            token_idx_map = {id(t): i for i, t in enumerate(tokens)}
            prev_label = NUM_CHUNK # Start token index (7)

            self.doc_meta[resume_id] = {
                "tokens": tokens
            }

            for chunk in _aggregate_chunks(tokens):
                chunk_tokens = chunk["tokens"]
                heading_parts = [t["token"] for t in chunk_tokens if t.get("bioLabel") in ("B-HEADING", "I-HEADING")]
                heading_text = " ".join(heading_parts) if heading_parts else chunk["heading"]
                heading_orig_idx = token_idx_map.get(id(chunk_tokens[0]), 0) if chunk_tokens else 0
                # NOTE: _merge_multiword_headings copies token dicts, so id(chunk_tokens[0])
                # usually misses token_idx_map → heading_orig_idx falls back to 0.
                # Browser inference must use spatial_all[0] to match this deployed behavior.
                # See training_pipeline/BROWSER_INFERENCE_PARITY.md §4.1
                heading_spatial = spatial[heading_orig_idx]

                sections = [
                    t.get("section") for t in chunk_tokens
                    if t.get("section") and t.get("section") != "NONE"
                ]
                if not sections:
                    continue
                section = Counter(sections).most_common(1)[0][0]
                if section not in CHUNK2ID:
                    continue

                is_virtual = chunk.get("virtual", False)

                if not is_virtual:
                    section = _normalize_chunk_label(heading_text, section)

                strip_heading = (split == "train" and random.random() < 0.25 and not is_virtual)

                if strip_heading:
                    body_tokens = [t for t in chunk_tokens if t.get("bioLabel", "O") not in ("B-HEADING", "I-HEADING")]
                    text_tokens = body_tokens if body_tokens else chunk_tokens
                    aug_idx = token_idx_map.get(id(text_tokens[0]), heading_orig_idx) if text_tokens else heading_orig_idx
                    heading_spatial = spatial[aug_idx]
                    text = " ".join(t["token"] for t in text_tokens if t["token"].strip())
                elif is_virtual:
                    text = "[SECTION_START] " + " ".join(t["token"] for t in chunk_tokens if t["token"].strip())
                else:
                    text = " ".join(t["token"] for t in chunk_tokens if t["token"].strip())

                enc = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].squeeze(0)
                attention_mask = enc["attention_mask"].squeeze(0)
                
                seq_len = input_ids.shape[0]
                if seq_len > self.max_length:
                    # Dynamically retain first 128 and last 128 tokens
                    input_ids = torch.cat([input_ids[:128], input_ids[-128:]], dim=0)
                    attention_mask = torch.cat([attention_mask[:128], attention_mask[-128:]], dim=0)
                else:
                    # Pad to max_length
                    pad_len = self.max_length - seq_len
                    input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)], dim=0)
                    attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)], dim=0)

                self.items.append({
                    "input_ids":        input_ids,
                    "attention_mask":   attention_mask,
                    "label":            torch.tensor(CHUNK2ID[section], dtype=torch.long),
                    "spatial_features": torch.tensor(heading_spatial,   dtype=torch.float32),
                    "doc_id":           resume_id,
                    "prev_label":       torch.tensor(prev_label, dtype=torch.long),
                    "heading":          heading_text,
                })
                prev_label = CHUNK2ID[section]

        print(f"[DATA] Loaded {len(self.items)} chunks from {len(files)} files.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]

def dataset_collate_fn(batch: list[dict]) -> dict:
    return {
        "input_ids":        torch.stack([b["input_ids"]        for b in batch]),
        "attention_mask":   torch.stack([b["attention_mask"]   for b in batch]),
        "labels":           torch.stack([b["label"]            for b in batch]),
        "spatial_features": torch.stack([b["spatial_features"] for b in batch]),
        "doc_ids":          [b["doc_id"]                       for b in batch],
        "prev_labels":      torch.stack([b["prev_label"]       for b in batch]),
        "headings":         [b["heading"]                      for b in batch],
    }
