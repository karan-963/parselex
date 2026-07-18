from __future__ import annotations

# ── Section labels (PI-0, legacy monolithic model) ────────────────────────────
SECTION_LABELS = ["NONE", "PERSONAL", "EXPERIENCE", "EDUCATION", "PROJECT", "SKILLS", "OTHER", "SUMMARY"]
SECTION2ID     = {l: i for i, l in enumerate(SECTION_LABELS)}
ID2SECTION     = {i: l for i, l in enumerate(SECTION_LABELS)}
NUM_SECTION    = len(SECTION_LABELS)

# ── Phase 1: Boundary Detection labels ───────────────────────────────────────
BOUNDARY_LABELS = ["O", "B-HEADING", "I-HEADING"]
BOUNDARY2ID     = {l: i for i, l in enumerate(BOUNDARY_LABELS)}
ID2BOUNDARY     = {i: l for i, l in enumerate(BOUNDARY_LABELS)}
NUM_BOUNDARY    = len(BOUNDARY_LABELS)

# ── Experience entry boundary detection labels ────────────────────────────────
EXP_BOUNDARY_LABELS = ["O", "B-ENTRY", "I-ENTRY"]
EXP_BOUNDARY2ID     = {l: i for i, l in enumerate(EXP_BOUNDARY_LABELS)}
ID2EXP_BOUNDARY     = {i: l for i, l in enumerate(EXP_BOUNDARY_LABELS)}
NUM_EXP_BOUNDARY    = len(EXP_BOUNDARY_LABELS)

# ── Phase 2: Section Chunk classification labels ──────────────────────────────
SECTION_CHUNK_LABELS = ["PERSONAL", "SUMMARY", "EXPERIENCE", "EDUCATION", "PROJECT", "SKILLS", "OTHER"]
CHUNK2ID             = {l: i for i, l in enumerate(SECTION_CHUNK_LABELS)}
ID2CHUNK             = {i: l for i, l in enumerate(SECTION_CHUNK_LABELS)}
NUM_CHUNK            = len(SECTION_CHUNK_LABELS)

# ── Per-model entity BIO labels ───────────────────────────────────────────────
_PERSONAL_TAGS = [
    "NAME", "EMAIL", "PHONE", "GITHUB", "LINKEDIN", "TWITTER",
    "LOCATION", "OTHER_LINK", "POSITION", "SUMMARY", "DOB", "HEADING",
]
_EXPERIENCE_TAGS = ["ROLE", "COMP", "COMP_LOC", "SDATE", "EDATE", "DESC", "HEADING"]
_EDUCATION_TAGS  = ["INST", "LOC", "DEG", "GPA", "INST_LOC", "DEGR", "SCORE", "SDATE", "EDATE", "DESC", "HEADING"]
_PROJECT_TAGS    = ["PROJ", "COMP", "SDATE", "EDATE", "DESC", "LOCATION", "HEADING"]
_SKILLS_TAGS     = ["SKILL_TYPE", "SKILL", "SKILL_CAT", "LANG_PRF", "DESC", "HEADING"]


def _build_bio(tags: list[str]) -> list[str]:
    out = ["O"]
    for t in tags:
        out += [f"B-{t}", f"I-{t}"]
    return out


def _build_bilou(tags: list[str]) -> list[str]:
    out = ["O"]
    for t in tags:
        out += [f"B-{t}", f"I-{t}", f"L-{t}", f"U-{t}"]
    return out


PERSONAL_BIO_MAP = {
    "O": 0, "B-NAME": 1, "I-NAME": 2, "B-PHONE": 3, "I-PHONE": 4,
    "B-EMAIL": 5, "I-EMAIL": 6, "B-LOCATION": 7, "I-LOCATION": 8,
    "B-POSITION": 9, "I-POSITION": 10, "B-LINKEDIN": 11, "I-LINKEDIN": 12,
    "B-GITHUB": 13, "B-OTHER_LINK": 14, "I-OTHER_LINK": 15, 
    "B-HEADING": 16, "I-HEADING": 17, "B-DOB": 18
}

ENTITY_LABELS: dict[str, list[str]] = {
    "personal":   [k for k, v in sorted(PERSONAL_BIO_MAP.items(), key=lambda x: x[1])],
    "experience": _build_bio(_EXPERIENCE_TAGS),
    "education":  _build_bilou(_EDUCATION_TAGS),
    "project":    _build_bilou(_PROJECT_TAGS),
    "skills":     _build_bilou(_SKILLS_TAGS),
}

ENTITY2ID: dict[str, dict[str, int]] = {
    k: {l: i for i, l in enumerate(v)} for k, v in ENTITY_LABELS.items()
}
ID2ENTITY: dict[str, dict[int, str]] = {
    k: {i: l for i, l in enumerate(v)} for k, v in ENTITY_LABELS.items()
}
NUM_ENTITY: dict[str, int] = {k: len(v) for k, v in ENTITY_LABELS.items()}

# ── Phase 3: Structural Phrase Segmentation labels ────────────────────────────
PHASE3_LABELS = ["O", "B-SEG", "I-SEG"]
NUM_PHASE3    = len(PHASE3_LABELS)

# ── Phase 4: Block Classification labels ──────────────────────────────────────
PHASE4_LABELS = ["ROLE", "COMP", "DATE", "DESC", "O"]
NUM_PHASE4    = len(PHASE4_LABELS)

# Projects Active Model Target Labels (Aligned with Experience for Transfer Learning)
# Index 0: DESC, Index 1: PROJ (Mirrors ROLE), Index 2: COMP, Index 3: SDATE, Index 4: EDATE
PROJ_CHUNK_LABELS = ['DESC', 'PROJ', 'COMP', 'SDATE', 'EDATE']



def num_labels(model_type: str) -> int:
    if model_type == "section":
        return NUM_SECTION
    if model_type == "boundary":
        return NUM_BOUNDARY
    if model_type == "section_chunk":
        return NUM_CHUNK
    if model_type in ("exp_boundary", "edu_boundary", "proj_boundary"):
        return NUM_EXP_BOUNDARY
    if model_type == "exp_label":
        return NUM_ENTITY["experience"]
    if model_type == "edu_label":
        return NUM_ENTITY["education"]
    if model_type == "proj_label":
        return NUM_ENTITY["project"]
    if model_type == "phase3_segmenter":
        return NUM_PHASE3
    if model_type == "proj_phase3_segmenter":
        return NUM_PHASE3       # same 3-class schema: O / B-SEG / I-SEG
    if model_type == "edu_phase3_segmenter":
        return NUM_PHASE3
    if model_type == "phase4_classifier":
        return NUM_PHASE4
    if model_type == "proj_phase4_classifier":
        return len(PROJ_CHUNK_LABELS)                # DESC / PROJ / COMP / SDATE / EDATE
    return NUM_ENTITY.get(model_type, NUM_ENTITY["personal"])


def label2id(model_type: str, label: str) -> int:
    if model_type == "section":
        return SECTION2ID.get(label, SECTION2ID["NONE"])
    return ENTITY2ID.get(model_type, ENTITY2ID["personal"]).get(label, 0)


def apply_sequence_constraints(logits: torch.Tensor, label_names: list[str]) -> torch.Tensor:
    """Apply sequential constraints to logits at runtime.
    
    logits shape: (SequenceLength, NumLabels) or (1, SequenceLength, NumLabels)
    """
    import torch
    # Clone to avoid in-place modification of model outputs
    logits_masked = logits.clone()
    
    # Handle optional batch dimension (usually 1 during inference)
    has_batch = len(logits_masked.shape) == 3
    if has_batch:
        B, L, C = logits_masked.shape
    else:
        L, C = logits_masked.shape
        B = 1
        logits_masked = logits_masked.unsqueeze(0)
        
    label2id = {name: i for i, name in enumerate(label_names)}
    i_indices = [i for i, name in enumerate(label_names) if name.startswith("I-")]
    
    if not i_indices:
        if not has_batch:
            logits_masked = logits_masked.squeeze(0)
        return logits_masked

    for b in range(B):
        # Step t = 0: cannot start with any I- tag
        logits_masked[b, 0, i_indices] -= 10000.0
        prev_pred_idx = logits_masked[b, 0].argmax().item()
        
        for t in range(1, L):
            prev_pred_name = label_names[prev_pred_idx] if prev_pred_idx < len(label_names) else "O"
            
            # Evaluate t-1 prediction name:
            if prev_pred_name == "O":
                logits_masked[b, t, i_indices] -= 10000.0
            elif prev_pred_name.startswith("B-") or prev_pred_name.startswith("I-"):
                active_tag = prev_pred_name[2:]
                for name, i in label2id.items():
                    if name.startswith("I-") and name[2:] != active_tag:
                        logits_masked[b, t, i] -= 10000.0
            else:
                logits_masked[b, t, i_indices] -= 10000.0
                
            prev_pred_idx = logits_masked[b, t].argmax().item()
            
    if not has_batch:
        logits_masked = logits_masked.squeeze(0)
        
    return logits_masked


def sort_tokens_by_reading_order(tokens: list[dict]) -> list[dict]:
    """Untangle interleaved tokens from multi-column layouts at runtime.
    
    Groups tokens by page and section, detects vertical gutters within each section,
    and sorts two-column sections column-by-column, while preserving single-column
    sections row-by-row.
    """
    from collections import defaultdict
    if not tokens:
        return []

    # Group by page and section
    page_section_tokens = defaultdict(list)
    for t in tokens:
        page = t["page"]
        section = t.get("section", "UNKNOWN")
        page_section_tokens[(page, section)].append(t)
        
    sorted_all_tokens = []
    
    # Process each page's groups in vertical order
    page_groups = defaultdict(list)
    for (page, section), group_tokens in page_section_tokens.items():
        avg_y0 = sum(t["y0"] for t in group_tokens) / len(group_tokens)
        page_groups[page].append(((page, section), avg_y0, group_tokens))
        
    for page in sorted(page_groups.keys()):
        # Sort groups on this page by average y0 ascending
        sorted_groups = sorted(page_groups[page], key=lambda x: x[1])
        
        for (page, section), _, pts in sorted_groups:
            # Skip two-column sorting for PERSONAL, SUMMARY, and SKILLS
            # instead, cluster them vertically and sort row-by-row (left to right)
            if section in {"PERSONAL", "SUMMARY", "SKILLS"}:
                pts_sorted_y = sorted(pts, key=lambda t: t["y0"])
                rows = []
                for t in pts_sorted_y:
                    placed = False
                    if rows:
                        last_row = rows[-1]
                        avg_y0 = sum(rt["y0"] for rt in last_row) / len(last_row)
                        avg_height = sum(rt["y1"] - rt["y0"] for rt in last_row) / len(last_row)
                        dynamic_tolerance = max(4.0, avg_height * 0.4)
                        if abs(t["y0"] - avg_y0) < dynamic_tolerance:
                            last_row.append(t)
                            placed = True
                    if not placed:
                        rows.append([t])
                
                sorted_section = []
                for line_idx, row in enumerate(rows):
                    row.sort(key=lambda t: t["x0"])
                    for t in row:
                        t_copy = dict(t)
                        t_copy["lineIndex"] = line_idx
                        sorted_section.append(t_copy)
                sorted_all_tokens.extend(sorted_section)
                continue
                
            xs = [t["x0"] for t in pts]
            x1s = [t["x1"] for t in pts]
            if not xs or not x1s:
                sorted_all_tokens.extend(pts)
                continue
                
            xmin, xmax = min(xs), max(x1s)
            width = xmax - xmin
            
            # Gutter detection scan
            scan_start = int(xmin + 0.20 * width)
            scan_end = int(xmin + 0.55 * width)
            
            best_x = None
            min_overlaps = float("inf")
            
            if scan_end > scan_start:
                for x in range(scan_start, scan_end + 1):
                    overlaps = 0
                    for t in pts:
                        if t["x0"] < x < t["x1"]:
                            overlaps += 1
                    if overlaps < min_overlaps:
                        min_overlaps = overlaps
                        best_x = x
            
            # Gutter verification
            has_gutter = False
            if best_x is not None and (min_overlaps <= 3 or min_overlaps < 0.08 * len(pts)):
                # Group section tokens by lineIndex
                line_tokens = defaultdict(list)
                for t in pts:
                    line_tokens[t["lineIndex"]].append(t)
                    
                crossing_lines = 0
                middle_gaps = 0
                for line_idx, lts in line_tokens.items():
                    # Check gap at best_x
                    lts_sorted = sorted(lts, key=lambda t: t["x0"])
                    overlap = False
                    for t in lts_sorted:
                        if t["x0"] < best_x < t["x1"]:
                            overlap = True
                            break
                    if overlap:
                        continue
                        
                    t_left = None
                    t_right = None
                    for t in lts_sorted:
                        if t["x1"] <= best_x:
                            if t_left is None or t["x1"] > t_left["x1"]:
                                t_left = t
                        if t["x0"] >= best_x:
                            if t_right is None or t["x0"] < t_right["x0"]:
                                t_right = t
                                
                    if t_left is not None and t_right is not None:
                        gap = t_right["x0"] - t_left["x1"]
                        if 10 <= gap <= 150:
                            middle_gaps += 1
                            
                # For a single section, require at least 2 middle-range gaps to confirm two-column layout
                if middle_gaps >= 2:
                    has_gutter = True
                    
            if has_gutter:
                g_mid = best_x
                left_list = []
                right_list = []
                
                for t in pts:
                    t_mid = (t["x0"] + t["x1"]) / 2
                    if t_mid < g_mid:
                        left_list.append(t)
                    else:
                        right_list.append(t)
                        
                left_list.sort(key=lambda t: (t["y0"], t["x0"]))
                right_list.sort(key=lambda t: (t["y0"], t["x0"]))
                
                sorted_section = left_list + right_list
            else:
                sorted_section = sorted(pts, key=lambda t: (t["lineIndex"], t["x0"]))
                
            sorted_all_tokens.extend(sorted_section)
            
    return sorted_all_tokens


def align_continuation_labels(tokens: list[dict], biolabs: list[str]) -> list[str]:
    """Stitch multi-line sentences by overriding B-X to I-X on line transitions."""
    if not tokens or len(tokens) != len(biolabs):
        return biolabs
        
    new_biolabs = list(biolabs)
    
    # Expanded bullet points list including common word bullet characters
    bullet_chars = {
        "•", "▪", "-", "*", "o", "■", "–", "—", "·", "", "", "✔", "▪", "➢", "",
        "\uf0a7", "\uf0d8", "\u2022", "\u2023", "\u2043", "\u254b", "\u25b8"
    }
    
    for i in range(1, len(tokens)):
        t_prev = tokens[i-1]
        t_curr = tokens[i]
        
        is_line_change = (t_curr.get("page") != t_prev.get("page")) or \
                         (t_curr.get("lineIndex") != t_prev.get("lineIndex"))
                         
        if is_line_change:
            prev_label = new_biolabs[i-1]
            curr_label = new_biolabs[i]
            
            if prev_label.startswith("B-") or prev_label.startswith("I-"):
                prev_entity = prev_label[2:]
                
                if curr_label == f"B-{prev_entity}":
                    prev_token_text = str(t_prev.get("token", "")).strip()
                    curr_token_text = str(t_curr.get("token", "")).strip()
                    
                    is_bullet = curr_token_text in bullet_chars or \
                                prev_token_text.endswith(":") or \
                                prev_token_text.endswith(";")
                                
                    if not is_bullet:
                        new_biolabs[i] = f"I-{prev_entity}"
                        
    return new_biolabs


def assign_heading_dist(tokens: list[dict]):
    """Assign distance-to-heading metrics based on original line sequence before sorting."""
    heading_lines = {}
    for t in tokens:
        if t.get("bioLabel") == "B-HEADING":
            page = t["page"]
            line = t["lineIndex"]
            if page not in heading_lines:
                heading_lines[page] = set()
            heading_lines[page].add(line)
            
    for page in heading_lines:
        heading_lines[page] = sorted(list(heading_lines[page]))
        
    for t in tokens:
        page = t["page"]
        line = t["lineIndex"]
        h_lines = heading_lines.get(page, [])
        h_dist = 99.0
        for hl in reversed(h_lines):
            if hl <= line:
                h_dist = float(line - hl)
                break
        t["heading_dist"] = h_dist

