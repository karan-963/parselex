"""
dataset.py
==========
PersonalSegmentDataset — the data module for the Personal/Meta segment classification pipeline.
"""

import os
import sys
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pymongo import MongoClient
from collections import Counter

# ─── Resolve generate_spacial_reports from the experience/phase3 directory ───────
_PERSONAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_PERSONAL_DIR)
_EXPERIENCE_PHASE3_DIR = os.path.join(
    _PIPELINE_DIR, "experience", "phase3_segment_classification"
)

# 1. Personal dir at index 0
if _PERSONAL_DIR in sys.path:
    sys.path.remove(_PERSONAL_DIR)
sys.path.insert(0, _PERSONAL_DIR)

# 2. Experience/phase3 at index 1 (behind personal/)
if _EXPERIENCE_PHASE3_DIR in sys.path:
    sys.path.remove(_EXPERIENCE_PHASE3_DIR)
sys.path.insert(1, _EXPERIENCE_PHASE3_DIR)

from generate_spacial_reports import clean_cid_tokens  # noqa: E402

# ─── Local config ────────────────────────────────────────────────────────────────
from config import (  # noqa: E402
    LABEL_LIST, LABEL2ID, ID2LABEL,
    EXCLUDED_RESUMES, is_personal_section_excluded,
    MONGO_URI, MONGO_DB,
    ENCODER_NAME, MAX_SEGS, MAX_SEG_LEN, SPATIAL_DIM,
)
from spatial_utils import extract_segment_spatial  # noqa: E402
from segment_heuristics import build_personal_segments, derive_segment_label  # noqa: E402

# ─── Bullet character set ────────────────────────────────────────────────────────
_BULLET_CHARS = {
    "•", "▪", "-", "*", "o", "■", "–", "—", "·", "", "", "✔",
    "▪", "➢", "", "\uf0a7", "\uf0d8", "\u2022", "\u2023",
    "\u2043", "\u254b", "\u25b8", "●",
}

# ─── Label Derivation ─────────────────────────────────────────────────────────────

def _get_segment_majority_section(seg_tokens: list[dict]) -> str:
    """Return the most common section label among the segment's constituent tokens."""
    sections = [t.get("section", "NONE") for t in seg_tokens if t.get("section")]
    if not sections:
        return "NONE"
    return Counter(sections).most_common(1)[0][0]

def is_personal_segment(segment: dict) -> bool:
    """True when the majority of the segment's tokens belong to the PERSONAL section."""
    return _get_segment_majority_section(segment.get("tokens", [])) == "PERSONAL"

def get_segment_class_label(segment: dict) -> str:
    """Derive segment class from token bioLabels (first-B rule via heuristics module)."""
    lbl = derive_segment_label(segment, LABEL_LIST)
    if lbl == "HETEROGENEOUS":
        return "O"
    return lbl

# ─── Dataset ──────────────────────────────────────────────────────────────────────

class PersonalSegmentDataset(Dataset):
    """
    Dataset for Personal/Meta Segment Classification.
    """

    def __init__(self, split: str = "train") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME, add_prefix_space=True)
        self.samples: list[dict] = []

        print(f"[DATA] Loading PersonalSegmentDataset for split='{split}' …")

        client = MongoClient(MONGO_URI)
        db     = client[MONGO_DB]

        query = {"tokens": {"$exists": True, "$ne": []}, "trainingMeta.split": split}
        docs  = list(db.resumes.find(query))
        client.close()

        print(f"[DATA] Retrieved {len(docs)} documents for split='{split}'.")

        label_freq: Counter = Counter()
        skipped_excluded = 0
        skipped_no_segs  = 0

        for doc in docs:
            resume_id = doc.get("resumeId", "unknown")

            # ── Contamination guard ───────────────────────────────────────────────
            if resume_id in EXCLUDED_RESUMES or is_personal_section_excluded(doc):
                print(f"[DATA]   Skipping excluded resume: {resume_id}")
                skipped_excluded += 1
                continue

            raw_tokens = doc.get("tokens", [])
            cleaned    = clean_cid_tokens(raw_tokens)
            cleaned.sort(key=lambda t: (t.get("page", 0), t.get("y0", 0.0), t.get("x0", 0.0)))
            personal_segs = build_personal_segments(cleaned)

            if not personal_segs:
                skipped_no_segs += 1
                continue

            seg_texts        : list[str]        = []
            spatial_features : list[list[float]] = []
            labels           : list[int]         = []

            for seg in personal_segs:
                lbl_str = get_segment_class_label(seg)
                lbl_id  = LABEL2ID.get(lbl_str, LABEL2ID["O"])
                seg_texts.append(seg["text"])
                spatial_features.append(
                    extract_segment_spatial(seg, all_tokens=cleaned, spatial_dim=SPATIAL_DIM)
                )
                labels.append(lbl_id)
                label_freq[LABEL_LIST[lbl_id]] += 1

            # ── Tokenise each segment text ────────────────────────────────────────
            seg_input_ids : list[torch.Tensor] = []
            seg_attn_mask : list[torch.Tensor] = []

            for text in seg_texts[:MAX_SEGS]:
                enc = self.tokenizer(
                    text,
                    max_length=MAX_SEG_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                seg_input_ids.append(enc["input_ids"].squeeze(0))
                seg_attn_mask.append(enc["attention_mask"].squeeze(0))

            # ── Pad / truncate to MAX_SEGS ────────────────────────────────────────
            num_segs = len(seg_texts)
            if num_segs < MAX_SEGS:
                pad_len  = MAX_SEGS - num_segs
                pad_tok  = torch.full((MAX_SEG_LEN,), self.tokenizer.pad_token_id, dtype=torch.long)
                pad_mask = torch.zeros(MAX_SEG_LEN, dtype=torch.long)
                for _ in range(pad_len):
                    seg_input_ids.append(pad_tok.clone())
                    seg_attn_mask.append(pad_mask.clone())
                    spatial_features.append([0.0] * SPATIAL_DIM)
                    labels.append(-100)

            seg_input_ids    = seg_input_ids[:MAX_SEGS]
            seg_attn_mask    = seg_attn_mask[:MAX_SEGS]
            spatial_features = spatial_features[:MAX_SEGS]
            labels           = labels[:MAX_SEGS]

            self.samples.append({
                "resume_id":       resume_id,
                "input_ids":       torch.stack(seg_input_ids),
                "attention_mask":  torch.stack(seg_attn_mask),
                "spatial_features": torch.tensor(spatial_features, dtype=torch.float32),
                "labels":          torch.tensor(labels, dtype=torch.long),
            })

        print(
            f"[DATA] PersonalSegmentDataset '{split}' — "
            f"{len(self.samples)} sequences built "
            f"| skipped_excluded={skipped_excluded} "
            f"| skipped_no_segs={skipped_no_segs}"
        )
        print(f"[DATA] Personal label distribution (active segments only): {dict(label_freq)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
