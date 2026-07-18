"""
dataset.py
==========
SkillsSegmentDataset — the canonical data module for the Skills token-level
sequence classification pipeline.

Architecture contract
---------------------
* Connects to MongoDB, retrieves resumes for a given split.
* Applies the EXCLUDED_RESUMES guard from config.py to prevent contamination.
* Uses `generate_spacial_reports.clean_cid_tokens` and
  `construct_sentences_by_appearance` (shared utility from the experience
  phase3 directory) to reconstruct visual line-segments from raw token streams.
* Tokenises each segment with distilroberta-base and computes a standard
  12-dimensional relative spatial feature vector per segment.
* For each segment, iterates element-by-element across the 32 subword token
  slots and fetches true token-level BIO targets using SKILLS_5CLASS_MAP /
  map_label_to_5class from data.py.
* Packs target labels into shape (MAX_SEGS, MAX_SEG_LEN). Special tokens
  (CLS, SEP) and empty trailing pad slots are masked with -100 to exclude
  them from the cross-entropy loss.
* Returns fixed-shape padded tensors compatible with the token-level
  SkillsSegmentClassifierModel (B × MAX_SEGS × MAX_SEG_LEN tensors).

Spatial feature vector (12D) — per segment, normalised:
  [0]  x0 / 612        – left edge (normalised to page width)
  [1]  y0 / 792        – top edge  (normalised to page height)
  [2]  x1 / 612        – right edge
  [3]  y1 / 792        – bottom edge
  [4]  w  / 612        – segment width
  [5]  h  / 792        – segment height
  [6]  fontSize / 30   – relative font size
  [7]  isBold          – binary bold flag (float)
  [8]  is_all_caps     – binary all-caps flag
  [9]  page / 10       – page index (normalised)
  [10] tier / 3        – indent tier (normalised)
  [11] has_bullet      – binary leading bullet flag

Label tensor (MAX_SEGS, MAX_SEG_LEN):
  -100  : CLS token, SEP token, padding tokens, non-SKILLS segments
  0–4   : 5-class BIO id from SKILLS_5CLASS_MAP for active SKILLS tokens
"""

import os
import sys
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pymongo import MongoClient
from collections import Counter

# ─── Resolve generate_spacial_reports from the experience/phase3 directory ───────
# Path strategy: skills/ at index 0, experience/phase3 at index 1.
# This ensures config / model / dataset always resolve to the local skills/ versions
# while generate_spacial_reports resolves to the experience/phase3 copy.
_SKILLS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_SKILLS_DIR)
_EXPERIENCE_PHASE3_DIR = os.path.join(
    _PIPELINE_DIR, "experience", "phase3_segment_classification"
)

# 1. Skills dir at index 0
if _SKILLS_DIR in sys.path:
    sys.path.remove(_SKILLS_DIR)
sys.path.insert(0, _SKILLS_DIR)

# 2. Experience/phase3 at index 1 (behind skills/)
if _EXPERIENCE_PHASE3_DIR in sys.path:
    sys.path.remove(_EXPERIENCE_PHASE3_DIR)
sys.path.insert(1, _EXPERIENCE_PHASE3_DIR)

from generate_spacial_reports import clean_cid_tokens, construct_sentences_by_appearance  # noqa: E402

# ─── Local config ────────────────────────────────────────────────────────────────
from config import (  # noqa: E402
    LABEL_LIST, LABEL2ID, ID2LABEL,
    EXCLUDED_RESUMES, is_skills_section_excluded,
    MONGO_URI, MONGO_DB,
    ENCODER_NAME, MAX_SEGS, MAX_SEG_LEN, SPATIAL_DIM,
)
from spatial_utils import extract_segment_spatial, extract_12d_spatial  # noqa: E402

# ─── 5-class BIO map and label mapper from data.py ──────────────────────────────
from data import SKILLS_5CLASS_MAP, map_label_to_5class  # noqa: E402


# ─── Bullet character set (mirrors generate_spacial_reports.BULLET_CHARS) ────────
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


def split_skills_segment(seg: dict) -> list[dict]:
    """Split segment into one segment per visual line if it spans multiple lines."""
    tokens = seg.get("tokens", [])
    if not tokens:
        return [seg]
        
    from collections import defaultdict
    line_toks = defaultdict(list)
    for t in tokens:
        key = (t.get("page", 0), t.get("lineIndex", 0))
        line_toks[key].append(t)
        
    if len(line_toks) <= 1:
        return [seg]
        
    split_segs = []
    for key in sorted(line_toks.keys()):
        toks = line_toks[key]
        toks.sort(key=lambda t: t.get("x0", 0.0))
        text = " ".join(t.get("token", "") for t in toks)
        
        x0 = min(t.get("x0", 0.0) for t in toks)
        y0 = min(t.get("y0", 0.0) for t in toks)
        x1 = max(t.get("x1", 0.0) for t in toks)
        y1 = max(t.get("y1", 0.0) for t in toks)
        fs = toks[0].get("fontSize", 9.0)
        is_bold = any(t.get("isBold", False) for t in toks)
        
        split_seg = {
            "text": text,
            "tokens": toks,
            "spatial": [fs, is_bold, 0.0, key[0], y0, x0, y1, x1, toks[0].get("tier", 0.0)]
        }
        split_segs.append(split_seg)
    return split_segs


def is_skills_segment(segment: dict) -> bool:
    """True when the majority of the segment's tokens belong to the SKILLS section."""
    return _get_segment_majority_section(segment.get("tokens", [])) == "SKILLS"


def is_heading_segment(segment: dict) -> bool:
    """True if the segment text or tokens indicate a section heading block."""
    text = (segment.get("text") or "").strip().lower()
    if ":" in text:
        return False
    text_clean = re.sub(r'[^a-z\s]', '', text).strip()
    
    heading_patterns = {
        "skills", "technical skills", "skill summary", "technologies", 
        "technology", "core competencies", "technical_skills", "competencies"
    }
    if text_clean in heading_patterns:
        return True
        
    seg_tok_list = segment.get("tokens", [])
    for tok in seg_tok_list:
        raw_bio = (tok.get("bioLabel") or "").upper()
        section = (tok.get("section") or "").upper()
        label = (tok.get("label") or "").upper()
        if "HEADING" in raw_bio or "HEADING" in label or section == "HEADING":
            return True
            
    return False


import re

def get_token_label_ids(
    seg_tokens: list[dict],
    tokenizer,
    segment_text: str,
    is_skills: bool,
) -> list[int]:
    """
    Produce a MAX_SEG_LEN-length list of token-level label IDs for one segment.

    Strategy
    --------
    1. Tokenise ``segment_text`` with distilroberta-base (max_length=MAX_SEG_LEN,
       padding="max_length"), obtaining subword token IDs and word_ids.
    2. For each subword position:
       - Position is CLS / SEP (word_id is None)  → -100  (masked)
       - Segment is not a SKILLS segment           → -100  (masked, outside loss)
       - First subword of a SKILLS token           → BIO label id from SKILLS_5CLASS_MAP
       - Continuation subword of same word         → -100  (masked, standard BIO practice)
       - Padding slot                              → -100  (masked)

    Parameters
    ----------
    seg_tokens   : raw token dicts from the segment (carrying ``bioLabel`` fields)
    tokenizer    : the shared AutoTokenizer instance
    segment_text : the reconstructed text string for this segment
    is_skills    : True if this segment belongs to the SKILLS section

    Returns
    -------
    list[int] of length MAX_SEG_LEN — label ids (0–4) or -100 for ignored positions.
    """
    enc = tokenizer(
        segment_text,
        max_length=MAX_SEG_LEN,
        padding="max_length",
        truncation=True,
        return_tensors=None,  # return plain python lists
    )
    word_ids = enc.word_ids()  # list of length MAX_SEG_LEN: int | None

    if not is_skills:
        # Entire segment is outside the SKILLS section — mask everything
        return [-100] * MAX_SEG_LEN

    # Build word-level BIO label ids from the segment's constituent token annotations.
    # We align by word index (position in the space-joined token sequence).
    # seg_tokens may differ slightly from the tokeniser's word count due to
    # space-joining, so we guard with an index-clamp.
    word_bio_ids: list[int] = []
    seg_clean = re.sub(r'[^A-Z\s]', '', segment_text.strip().upper()).strip()

    for tok in seg_tokens:
        token_text = (tok.get("token") or "").strip().upper()
        raw_bio = (tok.get("bioLabel", "O") or "O").upper()
        section = (tok.get("section", "") or "").upper()
        label = (tok.get("label", "") or "").upper()

        is_heading = (
            "HEADING" in raw_bio or
            "HEADING" in label or
            section == "HEADING" or
            token_text in {"SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES", "TECHNICAL_SKILLS"} or
            seg_clean in {"SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES", "TECHNICAL_SKILLS"}
        )

        if is_heading:
            word_bio_ids.append(-100)
        else:
            mapped  = map_label_to_5class(raw_bio)   # → "O" / "B-SKILL" / etc.
            word_bio_ids.append(SKILLS_5CLASS_MAP.get(mapped, 0))

    label_ids: list[int] = []
    prev_wid = None
    for wid in word_ids:
        if wid is None:
            # CLS or SEP token — always masked
            label_ids.append(-100)
        elif wid != prev_wid:
            # First subword of word `wid` — assign the word's BIO label
            if wid < len(word_bio_ids):
                label_ids.append(word_bio_ids[wid])
            else:
                label_ids.append(0)  # O for out-of-range (safety fallback)
        else:
            # Continuation subword — masked (standard for BIO token classification)
            label_ids.append(-100)
        prev_wid = wid

    return label_ids



# ─── Dataset ──────────────────────────────────────────────────────────────────────

class SkillsSegmentDataset(Dataset):
    """
    Dataset for Skills Token-Level Sequence Classification.

    Reads raw visual token data from MongoDB, reconstructs visual line-segments
    via the shared `generate_spacial_reports` utility, encodes each segment's
    text with MiniLM, and assembles the 16-D contract spatial vector per segment.

    The dataset yields fixed-shape padded sequences compatible with the
    token-level ``SkillsSegmentClassifierModel``:
        input_ids        : (MAX_SEGS, MAX_SEG_LEN)  LongTensor
        attention_mask   : (MAX_SEGS, MAX_SEG_LEN)  LongTensor
        spatial_features : (MAX_SEGS, SPATIAL_DIM)  FloatTensor
        labels           : (MAX_SEGS, MAX_SEG_LEN)  LongTensor  — -100 masked
    """

    def __init__(self, split: str = "train") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME, add_prefix_space=True)
        self.samples: list[dict] = []

        print(f"[DATA] Loading SkillsSegmentDataset for split='{split}' …")

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
            if resume_id in EXCLUDED_RESUMES or is_skills_section_excluded(doc):
                print(f"[DATA]   Skipping excluded resume: {resume_id}")
                skipped_excluded += 1
                continue

            raw_tokens = doc.get("tokens", [])
            cleaned    = clean_cid_tokens(raw_tokens)
            cleaned.sort(key=lambda t: (t.get("page", 0), t.get("y0", 0.0), t.get("x0", 0.0)))
            segments   = construct_sentences_by_appearance(cleaned)

            if not segments:
                skipped_no_segs += 1
                continue

            # ── Build per-segment text / spatial / token-label lists ──────────────
            seg_texts        : list[str]        = []
            spatial_features : list[list[float]] = []
            # labels_2d stores one list-of-int (length MAX_SEG_LEN) per segment
            labels_2d        : list[list[int]]  = []

            in_skills = False
            expanded_segments = []
            for seg in segments:
                text = (seg.get("text") or "").strip().lower()
                text_clean = re.sub(r'[^a-z\s]', '', text).strip()
                
                # Check for skills header or skills majority
                is_skills_hdr = text_clean in {
                    "skills", "technical skills", "skill summary", "technologies", 
                    "technology", "core competencies", "technical_skills", "competencies"
                }
                is_skills_maj = is_skills_segment(seg)
                
                if is_skills_hdr or is_skills_maj:
                    in_skills = True
                    
                # Check for exit via major structural headers without a colon
                MAJOR_STRUCTURAL_HEADERS = {
                    "experience", "work experience", "professional experience", "employment history",
                    "education", "academic background", "academics",
                    "projects", "academic projects", "key projects", "personal projects",
                    "summary", "professional summary", "career objective", "objective", "profile",
                    "personal details", "personal info", "personal information", "interests",
                    "certifications", "hobbies", "declarations", "declaration", "languages"
                }
                
                if in_skills and text_clean in MAJOR_STRUCTURAL_HEADERS and ":" not in text:
                    in_skills = False
                    
                if in_skills:
                    for sub_seg in split_skills_segment(seg):
                        expanded_segments.append((sub_seg, True))
                else:
                    expanded_segments.append((seg, False))

            for seg, skills in expanded_segments:
                if is_heading_segment(seg):
                    continue
                seg_tok_list = seg.get("tokens", [])

                # Derive token-level label ids for this segment's MAX_SEG_LEN slots
                tok_label_ids = get_token_label_ids(
                    seg_tokens   = seg_tok_list,
                    tokenizer    = self.tokenizer,
                    segment_text = seg["text"],
                    is_skills    = skills,
                )

                seg_texts.append(seg["text"])
                spatial_features.append(
                    extract_segment_spatial(seg, all_tokens=cleaned, spatial_dim=SPATIAL_DIM)
                )
                labels_2d.append(tok_label_ids)

                # Count active (non-masked) label tokens for reporting
                if skills:
                    for lid in tok_label_ids:
                        if lid != -100:
                            label_freq[LABEL_LIST[lid]] += 1

            # ── Tokenise each segment text into (MAX_SEG_LEN,) tensors ───────────
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
                seg_input_ids.append(enc["input_ids"].squeeze(0))      # (MAX_SEG_LEN,)
                seg_attn_mask.append(enc["attention_mask"].squeeze(0))  # (MAX_SEG_LEN,)

            # ── Pad / truncate all lists to MAX_SEGS ─────────────────────────────
            num_segs = len(seg_texts)
            if num_segs < MAX_SEGS:
                pad_len  = MAX_SEGS - num_segs
                pad_tok  = torch.full((MAX_SEG_LEN,), self.tokenizer.pad_token_id, dtype=torch.long)
                pad_mask = torch.zeros(MAX_SEG_LEN, dtype=torch.long)
                pad_lbl  = [-100] * MAX_SEG_LEN  # fully masked padding segment
                for _ in range(pad_len):
                    seg_input_ids.append(pad_tok.clone())
                    seg_attn_mask.append(pad_mask.clone())
                    spatial_features.append([0.0] * SPATIAL_DIM)
                    labels_2d.append(pad_lbl)

            seg_input_ids    = seg_input_ids[:MAX_SEGS]
            seg_attn_mask    = seg_attn_mask[:MAX_SEGS]
            spatial_features = spatial_features[:MAX_SEGS]
            labels_2d        = labels_2d[:MAX_SEGS]

            # ── Assemble final tensors ────────────────────────────────────────────
            # labels tensor shape: (MAX_SEGS, MAX_SEG_LEN) — token-level BIO ids
            self.samples.append({
                "resume_id":        resume_id,
                "input_ids":        torch.stack(seg_input_ids),                           # (MAX_SEGS, MAX_SEG_LEN)
                "attention_mask":   torch.stack(seg_attn_mask),                           # (MAX_SEGS, MAX_SEG_LEN)
                "spatial_features": torch.tensor(spatial_features, dtype=torch.float32),
                "labels":           torch.tensor(labels_2d, dtype=torch.long),            # (MAX_SEGS, MAX_SEG_LEN)
            })

        print(
            f"[DATA] SkillsSegmentDataset '{split}' — "
            f"{len(self.samples)} sequences built "
            f"| skipped_excluded={skipped_excluded} "
            f"| skipped_no_segs={skipped_no_segs}"
        )
        print(f"[DATA] Skills token class distribution (active positions only): {dict(label_freq)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
