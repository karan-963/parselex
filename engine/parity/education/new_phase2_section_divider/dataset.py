import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pymongo import MongoClient

from spatial_segments import clean_cid_tokens, construct_sentences_by_appearance

from config import (
    LABEL_LIST, LABEL2ID, ID2LABEL, GLOBAL_EXCLUSIONS, BACKBONE_NAME,
    is_education_section_excluded,
)
import boundary_config as bc
from education_label_alignment import (
    assign_education_segment_labels,
    resolve_education_boundary_heads,
)

def is_generic_role(text):
    clean_t = text.strip().rstrip(".:,;").strip().lower()
    generic_roles = {
        "software developer", "automation testing", "qa engineer",
        "software engineer", "web developer", "full stack developer", "frontend developer",
        "backend developer", "data scientist", "machine learning engineer", "android developer",
        "ios developer", "devops engineer", "associate software engineer", "senior software engineer",
        "lead software engineer", "qa analyst", "quality assurance engineer", "automation engineer",
        "java developer", "python developer", "net developer", "c# developer", "ui developer",
        "ux designer", "ui/ux designer", "frontend engineer", "backend engineer", "full stack engineer",
        "senior java developer", "senior python developer", "database administrator", "system administrator",
        "network engineer", "security engineer", "cloud architect", "solutions architect", "scrum master"
    }
    return clean_t in generic_roles

def is_tech_stack_leak(text):
    if ',' not in text and '|' not in text:
        return False
    tech_keywords = {
        "quantization", "convolutions", "apis", "frameworks", "tools", "languages",
        "technologies", "libraries", "databases", "python", "pytorch", "java", "c++",
        "javascript", "react", "html", "css", "sql", "git", "aws", "docker", "kubernetes",
        "tensorflow", "numpy", "pandas", "scikit", "rest", "graphql", "ci/cd", "agile",
        "scrum", "jira", "maven", "gradle", "jenkins", "terraform", "ansible", "linux"
    }
    has_tech_keyword = any(k in text.lower() for k in tech_keywords)
    if not has_tech_keyword:
        return False
    ACTIVE_VERBS = {
        "develop", "developed", "developing", "implement", "implemented", "implementing",
        "create", "created", "creating", "build", "built", "building", "design", "designed",
        "designing", "manage", "managed", "managing", "lead", "led", "leading", "optimize",
        "optimized", "optimizing", "write", "wrote", "writing", "integrate", "integrated",
        "integrating", "program", "programmed", "programming", "analyze", "analyzed", "analyzing"
    }
    core_nouns = {"system", "application", "platform", "website", "portal", "dashboard", "project", "model", "tool"}
    words = [w.strip().lower() for w in re.split(r'[,|]', text) if w.strip()]
    if not words:
        return False
    first_word = text.strip().split()[0].lower().rstrip(":,.-") if text.strip() else ""
    if first_word in ACTIVE_VERBS:
        return False
    if any(noun in text.lower() for noun in core_nouns):
        return False
    tech_count = sum(1 for w in words if any(k in w for k in tech_keywords) or len(w) < 15)
    return tech_count >= len(words) * 0.75

def get_segment_majority_section(seg_tokens):
    sections = [t.get("section", "NONE") for t in seg_tokens if t.get("section")]
    if not sections:
        return "NONE"
    from collections import Counter
    return Counter(sections).most_common(1)[0][0]

def is_education_segment(seg):
    """Return True if the segment's majority section is EDUCATION."""
    seg_tokens = seg.get("tokens", [])
    if not seg_tokens:
        return False
    majority = get_segment_majority_section(seg_tokens)
    return majority == "EDUCATION"

from education_spatial_builder import build_segment_spatial_features, pad_spatial_features
from education_train_weights import build_segment_loss_weights
from education_line_utils import build_physical_line_text_map

def group_segments_by_line(segments):
    """
    Groups segment indices by their physical line (page and lineIndex).
    This ensures that segments sharing the same horizontal line baseline 
    are grouped together, so that the first segment gets B-EDU_START and 
    subsequent inline segments get I-EDU_START.
    """
    num_segs = len(segments)
    groups   = []

    def get_line_key(seg):
        tokens = seg.get("tokens", [])
        for t in tokens:
            if t and "page" in t and "lineIndex" in t:
                return (t["page"], t["lineIndex"])
        return None

    i = 0
    while i < num_segs:
        curr_seg  = segments[i]
        curr_line = get_line_key(curr_seg)
        curr_group = [i]

        if curr_line is not None:
            j = i + 1
            while j < num_segs:
                next_seg = segments[j]
                next_line = get_line_key(next_seg)
                if next_line == curr_line:
                    curr_group.append(j)
                    j += 1
                else:
                    break
            i = j
        else:
            i += 1
            
        groups.append(curr_group)

    return groups


class SectionPhraseDataset(Dataset):
    def __init__(self, split: str = "train", max_segs: int = 128, max_seg_len: int = 32):
        self.tokenizer  = AutoTokenizer.from_pretrained(BACKBONE_NAME, add_prefix_space=True)
        self.max_segs   = max_segs
        self.max_seg_len = max_seg_len
        self.samples    = []

        print(f"[DATA] Loading Education SectionPhraseDataset for split '{split}'...")
        client = MongoClient(bc.MONGO_URI)
        db     = client[bc.MONGO_DB]

        query = {
            "tokens": {"$exists": True, "$ne": []},
            "trainingMeta.split": split,
            "educationEntryHeads": {"$exists": True, "$ne": []},
        }
        docs  = list(db.resumes.find(query))
        client.close()

        print(f"[DATA] Found {len(docs)} documents for split '{split}'. Processing phrase units...")

        for doc in docs:
            resume_id = doc.get("resumeId", "unknown")
            if resume_id in GLOBAL_EXCLUSIONS:
                print(f"[DATA] Skipping blacklisted/excluded resume ID: {resume_id}")
                continue

            if is_education_section_excluded(doc):
                continue

            raw_tokens = doc.get("tokens", [])
            cleaned    = clean_cid_tokens(raw_tokens)
            segments   = construct_sentences_by_appearance(cleaned)

            if not segments:
                continue

            groups = group_segments_by_line(segments)
            edu_heads = resolve_education_boundary_heads(doc, cleaned, segments)
            if not edu_heads:
                continue

            segment_labels = assign_education_segment_labels(segments, groups, edu_heads)
            physical_lines = build_physical_line_text_map(segments, cleaned)
            loss_weights = build_segment_loss_weights(
                segments, groups, segment_labels, physical_lines,
            )

            spatial_features = build_segment_spatial_features(
                segments, is_education_segment=is_education_segment, raw_tokens=cleaned,
            )

            seg_texts = [s["text"] for s in segments]

            # Section boundary masking: only EDUCATION segments carry live labels
            labels = []
            for i, s in enumerate(segments):
                if is_education_segment(s):
                    labels.append(LABEL2ID[segment_labels[i]])
                else:
                    labels.append(-100)

            seg_input_ids = []
            seg_attn_mask = []

            for text in seg_texts[:self.max_segs]:
                enc = self.tokenizer(
                    text,
                    max_length=self.max_seg_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                seg_input_ids.append(enc["input_ids"].squeeze(0))
                seg_attn_mask.append(enc["attention_mask"].squeeze(0))

            num_segs = len(seg_texts)
            if num_segs < self.max_segs:
                pad_len = self.max_segs - num_segs
                for _ in range(pad_len):
                    seg_input_ids.append(torch.full((self.max_seg_len,), self.tokenizer.pad_token_id, dtype=torch.long))
                    seg_attn_mask.append(torch.zeros(self.max_seg_len, dtype=torch.long))
                    labels.append(-100)
                    loss_weights.append(1.0)

            spatial_features = pad_spatial_features(spatial_features, self.max_segs)
            loss_weights = (loss_weights + [1.0] * self.max_segs)[:self.max_segs]

            seg_input_ids = seg_input_ids[:self.max_segs]
            seg_attn_mask = seg_attn_mask[:self.max_segs]
            labels = labels[:self.max_segs]

            self.samples.append({
                "resume_id":       resume_id,
                "input_ids":       torch.stack(seg_input_ids),
                "attention_mask":  torch.stack(seg_attn_mask),
                "spatial_features": torch.tensor(spatial_features, dtype=torch.float32),
                "labels":          torch.tensor(labels, dtype=torch.long),
                "loss_weights":    torch.tensor(loss_weights, dtype=torch.float32),
            })

        print(f"[DATA] Completed split '{split}'. Created {len(self.samples)} sample sequences.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
