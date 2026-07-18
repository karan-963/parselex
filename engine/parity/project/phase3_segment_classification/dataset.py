import os
import sys
import re
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pymongo import MongoClient

# Bootstrap path so we can import generate_spacial_reports from experience directory
PARENT_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIENCE_DIR = os.path.join(PARENT_DIR, "experience")
if EXPERIENCE_DIR not in sys.path:
    sys.path.append(EXPERIENCE_DIR)
from generate_spacial_reports import clean_cid_tokens, construct_sentences_by_appearance

from config import LABEL_LIST, LABEL2ID, ID2LABEL, EXCLUDED_RESUMES, BACKBONE_NAME
from segment_label_rules import refine_segment_label
from segment_context import enrich_spatial_with_project_context, build_sample_weights, build_hard_negative_mask

def get_segment_majority_section(seg_tokens):
    sections = [t.get("section", "NONE") for t in seg_tokens if t.get("section")]
    if not sections:
        return "NONE"
    from collections import Counter
    return Counter(sections).most_common(1)[0][0]


def is_project_segment(seg):
    """Return True if the majority section of this segment is PROJECT."""
    seg_tokens = seg.get("tokens", [])
    if not seg_tokens:
        return False
    if any(t.get("bioLabel") == "B-HEADING" for t in seg_tokens):
        return False
    return get_segment_majority_section(seg_tokens) == "PROJECT"


PROJECT_SECTION_KEY = "project"


def is_project_section_excluded(doc) -> bool:
    """True if this resume's Projects section is flagged out of training in MongoDB."""
    meta = doc.get("trainingMeta") or {}
    if meta.get("split") == "excluded":
        return True
    return PROJECT_SECTION_KEY in (doc.get("excludedSections") or [])


DATE_PATTERN = re.compile(
    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:\d{4}|\'\d{2}|\b\d{2})\b|'
    r'\b\d{1,2}\s*[/\-]\s*\d{4}\b|'
    r'\b\d{4}\s*[/\-]\s*\d{1,2}\b|'
    r'\b(?:present|date|till)\b',
    re.IGNORECASE
)

def split_hyphenated_segments(segments):
    new_segments = []
    for s in segments:
        text = s.get("text", "")
        if re.search(r'\s+[-—–]\s+', text):
            tokens = s.get("tokens", [])
            hyphen_idx = -1
            for idx, tok in enumerate(tokens):
                val = tok.get("token", "").strip()
                if val in ["-", "—", "–"]:
                    hyphen_idx = idx
                    break
            if hyphen_idx != -1:
                tokens_left = tokens[:hyphen_idx]
                tokens_right = tokens[hyphen_idx+1:]
                if tokens_left and tokens_right:
                    left_text = " ".join(t.get("token", "") for t in tokens_left).strip()
                    right_text = " ".join(t.get("token", "") for t in tokens_right).strip()
                    
                    is_left_date = bool(DATE_PATTERN.search(left_text))
                    is_right_date = bool(DATE_PATTERN.search(right_text))
                    
                    left_seg = {
                        "text": left_text,
                        "tokens": tokens_left,
                        "spatial": s["spatial"],
                        "_is_forced_proj": True,
                        "_is_split_left": True
                    }
                    right_seg = {
                        "text": right_text,
                        "tokens": tokens_right,
                        "spatial": s["spatial"],
                        "_is_forced_desc": True,
                        "_is_split_right": True
                    }
                    if is_left_date and is_right_date:
                        left_seg["_is_split_date_range"] = True
                        right_seg["_is_split_date_range"] = True
                        
                    new_segments.append(left_seg)
                    new_segments.append(right_seg)
                    continue
        new_segments.append(s)
    return new_segments

def get_segment_class_label(seg) -> str:
    """Derive the 4-class label for this segment based on raw bioLabel tokens."""
    seg_tokens = seg.get("tokens", [])
    
    # Check for date terms first (CRITICAL Date Safeguard)
    has_date = any(t.get("bioLabel", "O") in ["B-SDATE", "I-SDATE", "B-EDATE", "I-EDATE"] for t in seg_tokens)
    if has_date:
        return "DATE"
        
    clean_text = seg.get("text", "").strip()
    if DATE_PATTERN.search(clean_text):
        return "DATE"

    if seg.get("_is_forced_proj"):
        return refine_segment_label(clean_text, "PROJECT_NAME")
    if seg.get("_is_forced_desc"):
        return "DESC"
        
    # If no dates, we map tokens and use majority voting
    from collections import Counter
    votes = Counter()
    for t in seg_tokens:
        bio = t.get("bioLabel", "O")
        if bio in ["B-PROJ_NAME", "I-PROJ_NAME", "B-PROJ", "I-PROJ"]:
            votes["PROJECT_NAME"] += 1
        elif bio in ["B-SDATE", "I-SDATE", "B-EDATE", "I-EDATE"]:
            votes["DATE"] += 1
        else:
            votes["DESC"] += 1
            
    if not votes:
        return "DESC"

    raw_label = votes.most_common(1)[0][0]
    return refine_segment_label(clean_text, raw_label)


def extract_16d_spatial(s, prev_s=None, max_size=10.0, default_size=10.0, min_size=10.0):
    fs    = s["spatial"][0]
    bold  = float(s["spatial"][1])
    page  = float(s["spatial"][3])
    y0    = s["spatial"][4]
    x0    = s["spatial"][5]
    y1    = s["spatial"][6]
    x1    = s["spatial"][7]
    tier  = float(s["spatial"][8])

    text        = s["text"]
    is_all_caps = float(text.isupper() and len(text) > 2)
    bullets     = {"•", "*", "-", "▪", "◦", "■", "–", "—", "\uf0b7", "·", "✓", "✔", "\uf0a7", "●"}
    has_bullet  = float(any(text.startswith(b) for b in bullets))

    w = x1 - x0
    h = y1 - y0

    feat = [
        x0 / 612.0, y0 / 792.0, x1 / 612.0, y1 / 792.0,
        w  / 612.0, h  / 792.0,
        fs / 30.0,  bold, is_all_caps,
        page / 10.0, tier / 3.0, has_bullet
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

    # Feature Index 14 (Header Pattern Match): 1.0 if dominant style is bold and size == max_size; else 0.0
    feat_14 = 1.0 if (bold == 1.0 and abs(fs - max_size) < 1e-4) else 0.0

    # Feature Index 15 (Relative Tier Scalar)
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
    """Build inference-safe 16D spatial matrix with project-block context."""
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
    project_indices = [i for i, s in enumerate(segments) if is_project_segment(s)]
    return enrich_spatial_with_project_context(project_indices, segments, spatial)


class ProjectSegmentDataset(Dataset):
    def __init__(self, split: str = "train", max_segs: int = 128, max_seg_len: int = 32):
        self.tokenizer   = AutoTokenizer.from_pretrained(BACKBONE_NAME, add_prefix_space=True)
        self.max_segs    = max_segs
        self.max_seg_len = max_seg_len
        self.samples     = []

        print(f"[DATA] Loading Project SegmentDataset for split '{split}'...")
        client = MongoClient("mongodb://localhost:27017")
        db     = client["resume-labeling"]
        query  = {"tokens": {"$exists": True, "$ne": []}, "trainingMeta.split": split}
        docs   = list(db.resumes.find(query))
        client.close()

        print(f"[DATA] Found {len(docs)} documents for split '{split}'. Processing phrase units...")

        for doc in docs:
            resume_id = doc.get("resumeId", "unknown")
            if resume_id in EXCLUDED_RESUMES:
                continue
            if is_project_section_excluded(doc):
                continue

            raw_tokens = doc.get("tokens", [])
            cleaned    = clean_cid_tokens(raw_tokens)
            segments   = construct_sentences_by_appearance(cleaned)
            segments   = split_hyphenated_segments(segments)

            if not segments:
                continue

            # Data Leak Filtering: Discard resume if it has 0 positive ground truth project labels
            active_labels = [get_segment_class_label(s) for s in segments if is_project_segment(s)]
            positive_count = sum(1 for l in active_labels if l in ["PROJECT_NAME", "DATE"])
            if positive_count == 0:
                continue

            # Calculate project font size brackets strictly within PROJECT segments
            project_font_sizes = []
            for s in segments:
                if is_project_segment(s):
                    if "spatial" in s and len(s["spatial"]) > 0:
                        project_font_sizes.append(s["spatial"][0])
            if not project_font_sizes:
                project_font_sizes = [
                    s["spatial"][0]
                    for s in segments
                    if "spatial" in s and len(s["spatial"]) > 0
                ]
            if not project_font_sizes:
                project_font_sizes = [10.0]

            max_size = max(project_font_sizes)
            min_size = min(project_font_sizes)
            from collections import Counter
            c = Counter(project_font_sizes)
            default_size = c.most_common(1)[0][0]

            seg_texts        = [s["text"] for s in segments]
            spatial_features = build_spatial_feature_matrix(segments, max_size, default_size, min_size)

            # Section boundary masking: only PROJECT segments get live labels
            labels = []
            for s in segments:
                if is_project_segment(s):
                    labels.append(LABEL2ID[get_segment_class_label(s)])
                else:
                    labels.append(-100)

            sample_weights = build_sample_weights(labels, seg_texts)
            hard_negative_mask = build_hard_negative_mask(labels, seg_texts)

            seg_input_ids = []
            seg_attn_mask = []

            for text in seg_texts[:self.max_segs]:
                enc = self.tokenizer(
                    text, max_length=self.max_seg_len,
                    padding="max_length", truncation=True, return_tensors="pt"
                )
                seg_input_ids.append(enc["input_ids"].squeeze(0))
                seg_attn_mask.append(enc["attention_mask"].squeeze(0))

            num_segs = len(seg_texts)
            if num_segs < self.max_segs:
                for _ in range(self.max_segs - num_segs):
                    seg_input_ids.append(torch.full((self.max_seg_len,), self.tokenizer.pad_token_id, dtype=torch.long))
                    seg_attn_mask.append(torch.zeros(self.max_seg_len, dtype=torch.long))
                    spatial_features.append([0.0] * 16)
                    labels.append(-100)
                    sample_weights.append(0.0)
                    hard_negative_mask.append(0.0)

            seg_input_ids    = seg_input_ids[:self.max_segs]
            seg_attn_mask    = seg_attn_mask[:self.max_segs]
            spatial_features = spatial_features[:self.max_segs]
            labels           = labels[:self.max_segs]
            sample_weights   = sample_weights[:self.max_segs]
            hard_negative_mask = hard_negative_mask[:self.max_segs]

            self.samples.append({
                "resume_id":        resume_id,
                "input_ids":        torch.stack(seg_input_ids),
                "attention_mask":   torch.stack(seg_attn_mask),
                "spatial_features": torch.tensor(spatial_features, dtype=torch.float32),
                "labels":           torch.tensor(labels, dtype=torch.long),
                "sample_weights":   torch.tensor(sample_weights, dtype=torch.float32),
                "hard_negative_mask": torch.tensor(hard_negative_mask, dtype=torch.float32),
            })

        print(f"[DATA] Completed split '{split}'. Created {len(self.samples)} sample sequences.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
