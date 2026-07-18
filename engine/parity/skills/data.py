import os
import re
import torch
from torch.utils.data import Dataset
from pymongo import MongoClient
from transformers import AutoTokenizer
from collections import defaultdict, Counter

SKILLS_5CLASS_MAP = {
    "O": 0,
    "B-SKILL": 1,
    "I-SKILL": 2,
    "B-SKILL_TYPE": 3,
    "I-SKILL_TYPE": 4
}

def map_label_to_5class(label: str) -> str:
    """
    Maps any original skills label (including BILOU and other sections)
    to the target 5-class BIO schema:
    - O
    - B-SKILL, I-SKILL
    - B-SKILL_TYPE, I-SKILL_TYPE
    """
    if not label or label == "O":
        return "O"
    
    # Clean/normalize
    label = label.upper()
    
    # Parse prefix and entity tag
    if label.startswith("B-") or label.startswith("I-") or label.startswith("L-") or label.startswith("U-"):
        prefix = label[0]
        tag = label[2:]
    else:
        # Default fallback
        return "O"
    
    # Map entity tags
    if tag == "SKILL":
        target_tag = "SKILL"
    elif tag in ("SKILL_TYPE", "SKILL_CAT"):
        target_tag = "SKILL_TYPE"
    else:
        # Map HEADING, DESC, LANG_PRF, and anything else to O
        return "O"
        
    # Map BILOU prefix to BIO
    if prefix in ("B", "U"):
        return f"B-{target_tag}"
    elif prefix in ("I", "L"):
        return f"I-{target_tag}"
        
    return "O"

def clean_and_load_skills_tokens(split: str) -> list[dict]:
    """
    Connects to MongoDB, retrieves resumes for the specified split,
    runs cleaner/filters, and returns skills tokens for each resume.
    """
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGO_DB", "resume-labeling")
    
    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    
    query = {
        "tokensLoaded": True, 
        "trainingMeta.split": split
    }
    
    docs = list(db.resumes.find(
        query,
        {
            "resumeId": 1,
            "tokens": 1,
            "excludedSections": 1,
            "savedBy": 1,
            "_id": 0
        }
    ))
    client.close()
    
    cleaned_samples = []
    
    for doc in docs:
        resume_id = doc.get("resumeId", "unknown")
        excluded_sections = doc.get("excludedSections", [])
        if "skills" in excluded_sections:
            continue
            
        tokens = doc.get("tokens", [])
        if not tokens:
            continue
            
        # 1. Group by page and lineIndex
        line_map = defaultdict(list)
        for t in tokens:
            key = (t.get("page", 0), t.get("lineIndex", 0))
            line_map[key].append(t)
            
        # 2. Keep only lines that have at least one alphanumeric character
        valid_line_keys = set()
        for key, line_tokens in line_map.items():
            text = " ".join(t.get("token", "") for t in line_tokens)
            if re.search(r'[a-zA-Z0-9]', text):
                valid_line_keys.add(key)
                
        filtered_tokens = [t for t in tokens if (t.get("page", 0), t.get("lineIndex", 0)) in valid_line_keys]
        
        # 3. Clean cid:NNN font glyph tokens
        i = 0
        while i < len(filtered_tokens) - 2:
            t0 = filtered_tokens[i]
            t1 = filtered_tokens[i+1]
            t2 = filtered_tokens[i+2]
            if t0.get("token") == "cid" and t1.get("token") == ":" and t2.get("token", "").isdigit():
                t0["token"] = ""
                t1["token"] = ""
                t2["token"] = ""
                i += 3
            else:
                i += 1

        # Intercept section headings and force their bioLabel straight to "O"
        for t in filtered_tokens:
            token_text = (t.get("token") or "").strip().upper()
            raw_bio = (t.get("bioLabel", "O") or "O").upper()
            section = (t.get("section", "") or "").upper()
            label = (t.get("label", "") or "").upper()

            key = (t.get("page", 0), t.get("lineIndex", 0))
            line_tokens = line_map[key]
            line_text = " ".join(tok.get("token", "") for tok in line_tokens)
            line_clean = re.sub(r'[^A-Z\s]', '', line_text.strip().upper()).strip()

            is_heading = (
                "HEADING" in raw_bio or
                "HEADING" in label or
                section == "HEADING" or
                token_text in {"SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES", "TECHNICAL_SKILLS"} or
                line_clean in {"SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES", "TECHNICAL_SKILLS"}
            )
            if is_heading:
                t["bioLabel"] = "O"
                
        # 4. Extract skills section tokens
        skills_tokens = [
            t for t in filtered_tokens
            if t.get("section") == "SKILLS"
        ]
        
        # 5. Sort skills tokens by page, lineIndex, and x0
        skills_tokens.sort(key=lambda t: (t.get("page", 0), t.get("lineIndex", 0), t.get("x0", 0.0)))
        
        # 6. Aggressive Filtration: drop empty or whitespace-only tokens
        final_skills_tokens = []
        for t in skills_tokens:
            tok_text = t.get("token", "")
            if tok_text is not None and tok_text.strip() != "":
                final_skills_tokens.append(t)
                
        if len(final_skills_tokens) >= 3:
            cleaned_samples.append({
                "resumeId": resume_id,
                "tokens": final_skills_tokens,
                "savedBy": doc.get("savedBy", {})
            })
            
    return cleaned_samples

class SkillsDataset(Dataset):
    """
    Dataset for Skills Token Classification.
    Uses distilroberta-base and aligns words to subwords using the target 5-class schema.
    """
    def __init__(self, samples: list[dict], tokenizer_name: str = "distilroberta-base", max_length: int = 512):
        self.samples = samples
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
        self.lab2id = SKILLS_5CLASS_MAP
        self.id2lab = {v: k for k, v in self.lab2id.items()}
        
        self.items = []
        label_counts = Counter()
        
        for s in self.samples:
            resume_id = s["resumeId"]
            tokens = s["tokens"]
            
            words = [t["token"] for t in tokens]
            
            # Retrieve and map original labels to the target 5-class schema
            bio_labels = [map_label_to_5class(t.get("bioLabel", "O") or "O") for t in tokens]
            label_ids = [self.lab2id.get(lbl, 0) for lbl in bio_labels]
            
            for lbl in bio_labels:
                label_counts[lbl] += 1
                
            self.items.append({
                "words": words,
                "label_ids": label_ids,
                "doc_id": resume_id
            })
            
        print(f"[DATA] SkillsDataset — Loaded {len(self.items)} documents")
        print(f"[DATA] Token Class Frequencies: {dict(label_counts)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        words = item["words"]
        label_ids = item["label_ids"]
        doc_id = item["doc_id"]
        
        # Clean quotes and smart punctuation
        words_clean = [
            w.replace('"', '').replace("'", "").replace("‘", "").replace("’", "") 
            for w in words
        ]
        
        enc = self.tokenizer(
            words_clean,
            is_split_into_words=True,
            return_offsets_mapping=True,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        word_ids = enc.word_ids(0)
        
        # Align labels
        labels_out = []
        prev_wid = None
        
        for wid in word_ids:
            if wid is None:
                labels_out.append(-100)  # Mask out CLS, SEP, PAD tokens
            elif wid != prev_wid:
                # First subword gets the target label ID
                word_label = label_ids[wid] if wid < len(label_ids) else 0
                labels_out.append(word_label)
            else:
                # Trailing subwords are masked with -100
                labels_out.append(-100)
            prev_wid = wid
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels_out, dtype=torch.long),
            "doc_id": doc_id
        }

def skills_collate_fn(batch):
    """
    Custom collate function for batching variable-length sequence tensors.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    padded_inputs = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=1)
    padded_masks = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_ids": padded_inputs,
        "attention_mask": padded_masks,
        "labels": padded_labels,
        "doc_ids": [item["doc_id"] for item in batch]
    }
