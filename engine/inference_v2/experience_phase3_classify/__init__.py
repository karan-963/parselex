"""Experience training phase 3 — segment field classification (ROLE / COMP / DATE / DESC).

Takes the token stream after experience_phase2_divider has stamped B-ENTRY boundaries,
groups tokens into job entries, splits each entry into phrase blocks, and
classifies each block using ResumeChunkClassifier (MiniLM backbone).

Output per token: ``bioLabel`` is updated to B-ROLE/I-ROLE, B-COMP/I-COMP,
B-DATE/I-DATE, or B-DESC/I-DESC based on the block's predicted class.
"""

from __future__ import annotations

import os
import torch
from typing import Any
from transformers import AutoTokenizer

from .config import (
    MODEL_NAME, SPATIAL_DIM, NUM_LABELS, MAX_LENGTH,
    LABEL_LIST, PREV_LABEL_SENTINEL,
)
from .model import build_classifier
from .data_utils import (
    split_entry_blocks,
    merge_adjacent_date_blocks,
    group_experience_entries,
    extract_12d_spatial,
    is_punctuation_only,
    clean_block_text,
)
from .block_classification_report import build_block_classification_report
from ..confidence import max_prob
from ..model_precision import apply_precision
from ..predictor_cache import get_predictor


class PyTorchExperiencePhase2Predictor:
    """Loads the ResumeChunkClassifier checkpoint and runs block-level inference."""

    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )

        best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
        if not os.path.isfile(best_model_path):
            raise FileNotFoundError(f"Experience Phase 2 model not found: {best_model_path}")

        state_dict = torch.load(best_model_path, map_location=self.device, weights_only=True)

        # Infer spatial_dim from checkpoint
        spatial_key = "spatial_proj.mlp.0.weight"
        spatial_dim = state_dict[spatial_key].shape[1] if spatial_key in state_dict else SPATIAL_DIM

        # Infer num_labels from classifier head
        head_key = "classifier.head.weight"
        num_labels = state_dict[head_key].shape[0] if head_key in state_dict else NUM_LABELS

        self.model = build_classifier(
            num_labels=num_labels,
            spatial_dim=spatial_dim,
            model_name=MODEL_NAME,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()
        self.model, self.device = apply_precision(self.model, best_model_path, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, add_prefix_space=True, local_files_only=True
        )
        self.num_labels = num_labels
        self.spatial_dim = spatial_dim

    @torch.no_grad()
    def classify_blocks(
        self,
        blocks: list[list[dict]],
    ) -> tuple[list[int], list[float]]:
        """Classify a sequence of phrase blocks within one entry.

        Returns predicted label IDs and model softmax confidences (one per block).
        Uses the prev_label autoregressive context, starting with PREV_LABEL_SENTINEL.
        """
        predictions: list[int] = []
        confidences: list[float] = []
        prev_label_id = PREV_LABEL_SENTINEL  # warm-start sentinel

        for block in blocks:
            text = clean_block_text(block).replace('"', "")
            if not text or is_punctuation_only(text):
                predictions.append(0)  # default DESC for noise
                confidences.append(1.0)
                continue

            spatial_vec = extract_12d_spatial(block[0], text)

            enc = self.tokenizer(
                text,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            spatial_t = torch.tensor(
                [spatial_vec], dtype=torch.float32, device=self.device
            )  # (1, spatial_dim)
            prev_labels_t = torch.tensor(
                [prev_label_id], dtype=torch.long, device=self.device
            )

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                spatial_features=spatial_t,
                prev_labels=prev_labels_t,
            )
            logits = out["logits"]
            pred_id = int(logits.argmax(-1).item())
            predictions.append(pred_id)
            confidences.append(max_prob(logits, pred_id))
            prev_label_id = pred_id

        return predictions, confidences


# ── Contextual override heuristics ─────────────────────────────────────────────

ROLE_KEYWORDS = {
    "developer", "engineer", "consultant", "analyst", "architect", "manager",
    "designer", "lead", "associate", "intern", "specialist", "director",
    "programmer", "administrator", "coordinator", "executive", "officer",
    "head", "founder", "trainee", "apprentice", "graduate", "assistant",
    "researcher", "scientist", "technician", "operator", "supervisor",
    "contractor", "freelancer", "vp", "president", "cto", "ceo", "coo",
    "tester", "expert",
}

COMP_KEYWORDS = {
    "pvt", "ltd", "inc", "corp", "corporation", "limited", "solutions",
    "services", "technologies", "university", "college", "school", "bank",
    "systems", "labs", "consulting", "company", "institutions", "association",
}


def _apply_overrides(blocks: list[list[dict]], predictions: list[int]) -> list[int]:
    """Apply keyword / length heuristics to fix obvious classification errors."""
    in_tech_stack_context = False

    for i, block in enumerate(blocks):
        text = clean_block_text(block).strip()
        text_lower = text.lower()
        word_count = len(text_lower.split())

        is_tech_list = (
            "technology stack" in text_lower
            or "tech stack" in text_lower
            or text_lower.startswith("technology :")
            or text_lower.startswith("tech :")
        )
        if is_tech_list:
            in_tech_stack_context = True

        if word_count > 6 and predictions[i] == 0:
            in_tech_stack_context = False

        if predictions[i] in (1, 2):  # ROLE or COMP
            words = [w for w in text.split() if any(c.isalpha() for c in w)]
            cap_words = [w for w in words if w[0].isupper()]
            is_proper_noun = (len(cap_words) / len(words) >= 0.7) if words else False
            has_keyword = any(w in text_lower for w in ROLE_KEYWORDS | COMP_KEYWORDS)

            if (
                is_tech_list
                or in_tech_stack_context
                or word_count > 10
                or (word_count > 4 and not (has_keyword or is_proper_noun))
            ):
                predictions[i] = 0  # force to DESC

    return predictions


# ── Public entry point ─────────────────────────────────────────────────────────

def run_experience_phase3_classify(
    tokens: list[dict],
    resume_id: str = "resume",
    *,
    slug: str | None = None,
) -> dict[str, Any]:
    """Classify experience tokens into ROLE/COMP/DATE/DESC blocks.

    Reads the B-ENTRY boundary labels written by experience_phase2_divider and groups
    the experience tokens into entries. Each entry is split into phrase blocks
    which are classified one by one.

    Writes ``bioLabel`` back to every token in the main list.
    Returns a result dict compatible with the inference pipeline artifact format.
    """
    # 1. Collect experience tokens (excluding headings)
    filtered_indices = [
        i for i, t in enumerate(tokens)
        if t.get("section") == "EXPERIENCE"
        and t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
    ]

    if len(filtered_indices) < 3:
        return _empty_result(tokens, filtered_indices, resume_id)

    filtered_tokens = [tokens[i] for i in filtered_indices]

    # 2. Primary entry slice heads (same as step 9 — not every scattered B-ENTRY token)
    from inference_v2.experience_phase1_segment.entry_slice_heads import resolve_entry_slice_heads

    b_entry_lines = resolve_entry_slice_heads(filtered_tokens)
    if not b_entry_lines:
        b_entry_lines = {
            (t.get("page"), t.get("lineIndex", t.get("line_index", 0)))
            for t in filtered_tokens
            if (t.get("token") or "").strip() in {"•", "●"}
        }

    if not b_entry_lines:
        # No B-ENTRY boundaries found from phase1 — leave bioLabels as-is and output O.
        # Do NOT overwrite tokens — phase1 results should be preserved.
        return _empty_result(tokens, filtered_indices, resume_id,
                             reason="no_b_entry_boundaries")

    # 3. Group into entries using boundary lines
    entries = group_experience_entries(filtered_tokens, b_entry_lines)
    if not entries:
        return _empty_result(tokens, filtered_indices, resume_id,
                             reason="no_entries_formed")

    # 4. Load model
    predictor = get_predictor("experience_phase3_classify", PyTorchExperiencePhase2Predictor)

    # 5. Per-entry block classification
    final_labels: dict[int, str] = {}  # token id → bio label string
    report_block_rows: list[dict] = []

    for head, entry_toks in entries:
        if len(entry_toks) < 2:
            continue

        raw_blocks = split_entry_blocks(entry_toks)
        blocks = merge_adjacent_date_blocks(raw_blocks)
        if not blocks:
            continue

        predictions, confidences = predictor.classify_blocks(blocks)
        for block, conf in zip(blocks, confidences):
            for tok in block:
                tok["confidence"] = conf
        predictions = _apply_overrides(blocks, predictions)

        entry_key = f"JOB p{head[0]} L{head[1]}"
        for block, pred_id, conf in zip(blocks, predictions, confidences):
            label_str = LABEL_LIST[pred_id]
            text = clean_block_text(block)
            if text and not is_punctuation_only(text):
                report_block_rows.append({
                    "entry_key": entry_key,
                    "text": text,
                    "pred": label_str,
                    "confidence": conf,
                    "block": block,
                })

            for j, t in enumerate(block):
                bio = f"B-{label_str}" if j == 0 else f"I-{label_str}"
                final_labels[id(t)] = bio

    block_report = build_block_classification_report(
        resume_id, report_block_rows, filtered_tokens, slug
    )

    # 6. Write predictions back to main tokens list
    for idx in filtered_indices:
        t = tokens[idx]
        pred_bio = final_labels.get(id(t), "O")
        t["bioLabel"] = pred_bio
        t["bio_label"] = pred_bio

    non_o_count = sum(1 for idx in filtered_indices if tokens[idx]["bioLabel"] != "O")

    return {
        "stage": "experience_phase3_classify",
        "title": "Experience Field Classification",
        "section": "EXPERIENCE",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "experience/phase3_segment_classification",
        "task": "field_classification",
        "labelField": "prediction",
        "labels": LABEL_LIST,
        "blockClassification": block_report,
        "gtEntryHeadSource": "mongodb.experienceEntryHeads",
        "tokenCount": len(filtered_indices),
        "evalTokenCount": len(filtered_indices),
        "nonOCount": non_o_count,
        "sampleLabels": sorted(set(
            tokens[idx]["bioLabel"] for idx in filtered_indices
        )),
        "tokens": [
            {
                "page": tokens[idx]["page"],
                "lineIndex": tokens[idx]["lineIndex"],
                "tokenIndex": tokens[idx]["tokenIndex"],
                "token": tokens[idx]["token"],
                "prediction": tokens[idx]["bioLabel"],
                "confidence": tokens[idx].get("confidence", 0.0),
                "x0": tokens[idx].get("x0"),
                "y0": tokens[idx].get("y0"),
                "x1": tokens[idx].get("x1"),
                "y1": tokens[idx].get("y1"),
            }
            for idx in filtered_indices
        ],
    }


def _empty_result(
    tokens: list[dict],
    filtered_indices: list[int],
    resume_id: str,
    reason: str = "empty",
) -> dict[str, Any]:
    return {
        "stage": "experience_phase3_classify",
        "title": "Experience Field Classification",
        "section": "EXPERIENCE",
        "resumeId": resume_id,
        "source": "training-engine/inference_v2",
        "trainingPipeline": "experience/phase3_segment_classification",
        "labelField": "bioLabel",
        "labels": LABEL_LIST,
        "tokenCount": len(filtered_indices),
        "evalTokenCount": len(filtered_indices),
        "nonOCount": 0,
        "sampleLabels": [],
        "tokens": [
            {
                "page": tokens[idx]["page"],
                "lineIndex": tokens[idx]["lineIndex"],
                "tokenIndex": tokens[idx]["tokenIndex"],
                "token": tokens[idx]["token"],
                "prediction": "O",
                "x0": tokens[idx].get("x0"),
                "y0": tokens[idx].get("y0"),
                "x1": tokens[idx].get("x1"),
                "y1": tokens[idx].get("y1"),
            }
            for idx in filtered_indices
        ],
    }
