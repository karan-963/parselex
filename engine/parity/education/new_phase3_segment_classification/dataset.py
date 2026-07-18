import os
import re
import sys

import torch
from pymongo import MongoClient
from torch.utils.data import Dataset
from transformers import AutoTokenizer

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIENCE_DIR = os.path.join(PARENT_DIR, "experience")
if EXPERIENCE_DIR not in sys.path:
    sys.path.append(EXPERIENCE_DIR)
from generate_spacial_reports import clean_cid_tokens, construct_sentences_by_appearance

from config import (
    BACKBONE_NAME,
    GLOBAL_EXCLUSIONS,
    LABEL2ID,
    LABEL_LIST,
    is_education_section_excluded,
)
from segment_context import (
    build_hard_negative_mask,
    build_sample_weights,
    enrich_spatial_with_education_context,
)
from segment_label_rules import is_education_section_heading, refine_segment_label

_BIO_TO_CLASS = {
    "B-INST": "INSTITUTION",
    "I-INST": "INSTITUTION",
    "B-DEG": "DEGREE",
    "I-DEG": "DEGREE",
    "B-SDATE": "DATE",
    "I-SDATE": "DATE",
    "B-EDATE": "DATE",
    "I-EDATE": "DATE",
}

DATE_PATTERN = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:\d{4}|\'\d{2}|\b\d{2})\b|"
    r"\b\d{1,2}\s*[/\-]\s*\d{4}\b|"
    r"\b\d{4}\s*[/\-]\s*\d{1,2}\b|"
    r"\b(?:19|20)\d{2}\b|"
    r"\b(?:present|till)\b",
    re.IGNORECASE,
)


def get_segment_majority_section(seg_tokens):
    sections = [t.get("section", "NONE") for t in seg_tokens if t.get("section")]
    if not sections:
        return "NONE"
    from collections import Counter
    return Counter(sections).most_common(1)[0][0]


def is_education_segment(seg):
    """True for EDUCATION-region content segments; excludes section heading lines."""
    seg_tokens = seg.get("tokens", [])
    if not seg_tokens:
        return False
    if is_education_section_heading(seg):
        return False
    return get_segment_majority_section(seg_tokens) == "EDUCATION"


def get_segment_class_label(seg) -> str:
    seg_tokens = seg.get("tokens", [])
    has_date = any(
        t.get("bioLabel", "O") in ["B-SDATE", "I-SDATE", "B-EDATE", "I-EDATE"]
        for t in seg_tokens
    )
    if has_date:
        return "DATE"

    clean_text = seg.get("text", "").strip()
    if DATE_PATTERN.search(clean_text):
        return "DATE"

    from collections import Counter
    bio_votes = Counter()
    for t in seg_tokens:
        bio = t.get("bioLabel", "O")
        bio_votes[bio] += 1
    if not bio_votes:
        return "DESCRIPTION"
    majority_bio = bio_votes.most_common(1)[0][0]
    raw_label = _BIO_TO_CLASS.get(majority_bio, "DESCRIPTION")
    return refine_segment_label(clean_text, raw_label)


def extract_16d_spatial(s, prev_s=None, max_size=10.0, default_size=10.0, min_size=10.0):
    fs = s["spatial"][0]
    bold = float(s["spatial"][1])
    page = float(s["spatial"][3])
    y0 = s["spatial"][4]
    x0 = s["spatial"][5]
    y1 = s["spatial"][6]
    x1 = s["spatial"][7]
    tier = float(s["spatial"][8])

    text = s["text"]
    is_all_caps = float(text.isupper() and len(text) > 2)
    bullets = {"•", "*", "-", "▪", "◦", "■", "–", "—", "\uf0b7", "·", "✓", "✔", "\uf0a7", "●"}
    has_bullet = float(any(text.startswith(b) for b in bullets))

    w = x1 - x0
    h = y1 - y0

    feat = [
        x0 / 612.0, y0 / 792.0, x1 / 612.0, y1 / 792.0,
        w / 612.0, h / 792.0,
        fs / 30.0, bold, is_all_caps,
        page / 10.0, tier / 3.0, has_bullet,
    ]

    if prev_s:
        fs_prev = prev_s["spatial"][0]
        bold_prev = float(prev_s["spatial"][1])
        page_prev = float(prev_s["spatial"][3])
        y1_prev = prev_s["spatial"][6]

        if page == page_prev:
            font_tier_delta = 1.0 if (fs < fs_prev or (bold_prev == 1.0 and bold == 0.0)) else 0.0
            visual_spacing_gap = (y0 - y1_prev) / 792.0
        else:
            font_tier_delta = 0.0
            visual_spacing_gap = 0.0
    else:
        font_tier_delta = 0.0
        visual_spacing_gap = 0.0

    feat.append(font_tier_delta)
    feat.append(visual_spacing_gap)

    feat_14 = 1.0 if (bold == 1.0 and abs(fs - max_size) < 1e-4) else 0.0
    diff_max = abs(fs - max_size)
    diff_default = abs(fs - default_size)
    diff_min = abs(fs - min_size)
    min_diff = min(diff_max, diff_default, diff_min)
    if min_diff == diff_max:
        feat_15 = 1.0
    elif min_diff == diff_default:
        feat_15 = 0.5
    else:
        feat_15 = 0.0

    feat.append(feat_14)
    feat.append(feat_15)
    return feat


def build_spatial_feature_matrix(
    segments: list[dict],
    max_size: float = 10.0,
    default_size: float = 10.0,
    min_size: float = 10.0,
) -> list[list[float]]:
    spatial = [
        extract_16d_spatial(
            segments[i],
            segments[i - 1] if i > 0 else None,
            max_size,
            default_size,
            min_size,
        )
        for i in range(len(segments))
    ]
    education_indices = [i for i, s in enumerate(segments) if is_education_segment(s)]
    return enrich_spatial_with_education_context(education_indices, segments, spatial)


class EducationSegmentDataset(Dataset):
    def __init__(self, split: str = "train", max_segs: int = 128, max_seg_len: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME, add_prefix_space=True)
        self.max_segs = max_segs
        self.max_seg_len = max_seg_len
        self.samples = []

        print(f"[DATA] Loading Education SegmentDataset for split '{split}'...")
        client = MongoClient("mongodb://localhost:27017")
        db = client["resume-labeling"]
        query = {"tokens": {"$exists": True, "$ne": []}, "trainingMeta.split": split}
        docs = list(db.resumes.find(query))
        client.close()

        print(f"[DATA] Found {len(docs)} documents for split '{split}'. Processing phrase units...")

        for doc in docs:
            resume_id = doc.get("resumeId", "unknown")
            if resume_id in GLOBAL_EXCLUSIONS:
                continue
            if is_education_section_excluded(doc):
                continue

            raw_tokens = doc.get("tokens", [])
            cleaned = clean_cid_tokens(raw_tokens)
            segments = construct_sentences_by_appearance(cleaned)

            if not segments:
                continue

            active_labels = [get_segment_class_label(s) for s in segments if is_education_segment(s)]
            positive_count = sum(1 for lbl in active_labels if lbl in ["INSTITUTION", "DEGREE", "DATE"])
            if positive_count == 0:
                continue

            edu_font_sizes = []
            for s in segments:
                if is_education_segment(s) and s.get("spatial"):
                    edu_font_sizes.append(s["spatial"][0])
            if not edu_font_sizes:
                edu_font_sizes = [
                    s["spatial"][0] for s in segments if s.get("spatial") and len(s["spatial"]) > 0
                ]
            if not edu_font_sizes:
                edu_font_sizes = [10.0]

            from collections import Counter
            max_size = max(edu_font_sizes)
            min_size = min(edu_font_sizes)
            default_size = Counter(edu_font_sizes).most_common(1)[0][0]

            seg_texts = [s["text"] for s in segments]
            spatial_features = build_spatial_feature_matrix(segments, max_size, default_size, min_size)

            labels = []
            for s in segments:
                if is_education_segment(s):
                    labels.append(LABEL2ID[get_segment_class_label(s)])
                else:
                    labels.append(-100)

            sample_weights = build_sample_weights(labels, seg_texts)
            hard_negative_mask = build_hard_negative_mask(labels, seg_texts)

            seg_input_ids = []
            seg_attn_mask = []

            for text in seg_texts[: self.max_segs]:
                enc = self.tokenizer(
                    text,
                    max_length=self.max_seg_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                seg_input_ids.append(enc["input_ids"].squeeze(0))
                seg_attn_mask.append(enc["attention_mask"].squeeze(0))

            num_segs = len(seg_texts)
            if num_segs < self.max_segs:
                for _ in range(self.max_segs - num_segs):
                    seg_input_ids.append(
                        torch.full((self.max_seg_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                    )
                    seg_attn_mask.append(torch.zeros(self.max_seg_len, dtype=torch.long))
                    spatial_features.append([0.0] * 16)
                    labels.append(-100)
                    sample_weights.append(0.0)
                    hard_negative_mask.append(0.0)

            seg_input_ids = seg_input_ids[: self.max_segs]
            seg_attn_mask = seg_attn_mask[: self.max_segs]
            spatial_features = spatial_features[: self.max_segs]
            labels = labels[: self.max_segs]
            sample_weights = sample_weights[: self.max_segs]
            hard_negative_mask = hard_negative_mask[: self.max_segs]

            self.samples.append({
                "resume_id": resume_id,
                "input_ids": torch.stack(seg_input_ids),
                "attention_mask": torch.stack(seg_attn_mask),
                "spatial_features": torch.tensor(spatial_features, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
                "sample_weights": torch.tensor(sample_weights, dtype=torch.float32),
                "hard_negative_mask": torch.tensor(hard_negative_mask, dtype=torch.float32),
            })

        print(f"[DATA] Completed split '{split}'. Created {len(self.samples)} sample sequences.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
