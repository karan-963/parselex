import os
import re
from collections import defaultdict
from pymongo import MongoClient

# Database configuration (matching config.py)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "resume-labeling")

BULLET_CHARS = {
    "•", "*", "-", "▪", "◦", "■", "–", "—", "\uf0b7", "·", "✓", "✔", "\uf0a7", "●", "❖", "o", "", "", "➢", "\uf0d8", "\u2022", "\u2023", "\u2043", "\u254b", "\u25b8"
}

def clean_cid_tokens(tokens):
    """Strip out layout artifacts like cid:NNN."""
    clean = []
    j = 0
    while j < len(tokens):
        t0 = tokens[j]
        if (j + 2 < len(tokens)
                and t0.get("token") == "cid"
                and tokens[j + 1].get("token") == ":"
                and (tokens[j + 2].get("token") or "").isdigit()):
            j += 3
        else:
            clean.append(t0)
            j += 1
    return clean

def get_dominant_font_info(line_tokens):
    """Return dominant (median) font size and bold status of a line/sub-line based on token majority."""
    if not line_tokens:
        return 9.0, False
    font_sizes = [t.get("fontSize", 9.0) for t in line_tokens]
    font_sizes.sort()
    median_fs = font_sizes[len(font_sizes)//2]
    
    bold_count = sum(1 for t in line_tokens if t.get("isBold", False))
    is_bold = bold_count > (len(line_tokens) / 2)
    return median_fs, is_bold

def build_line_record(page, line_idx, line_tokens):
    """
    Builds a unified line record where properties like font size and bold status 
    are derived strictly from the majority vote of the constituent tokens.
    """
    text = " ".join(t.get("token", "") for t in line_tokens).strip()
    x0 = min(t.get("x0", 0.0) for t in line_tokens)
    y0 = min(t.get("y0", 0.0) for t in line_tokens)
    x1 = max(t.get("x1", 0.0) for t in line_tokens)
    y1 = max(t.get("y1", 0.0) for t in line_tokens)
    
    # Majority vote for font size and bold status
    font_sizes = [t.get("fontSize", 9.0) for t in line_tokens]
    bold_states = [t.get("isBold", False) for t in line_tokens]
    
    dominant_fs = max(set(font_sizes), key=font_sizes.count) if font_sizes else 9.0
    dominant_bold = max(set(bold_states), key=bold_states.count) if bold_states else False
    
    # Determine the content_x0 (x0 excluding bullet tokens)
    content_x0 = x0
    if len(line_tokens) > 1 and line_tokens[0].get("token", "").strip() in BULLET_CHARS:
        content_x0 = line_tokens[1].get("x0", x0)
        
    return {
        "page": page,
        "lineIndex": line_idx,
        "tokens": line_tokens,
        "text": text,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "fontSize": dominant_fs,
        "isBold": dominant_bold,
        "content_x0": content_x0
    }

def construct_sentences_by_appearance(tokens, allow_vertical_merge=False):
    """
    Groups raw layout tokens into distinct visual rows, resolves horizontal metadata 
    splits, and aggregates rows into continuous contextual paragraphs.
    """
    tokens = [t for t in tokens if t.get("token", "").strip()]
    if not tokens:
        return []

    # Calculate median font size for prominence checking
    font_sizes = [t.get("fontSize", 9.0) for t in tokens]
    font_sizes.sort()
    median_font_size = font_sizes[len(font_sizes) // 2] if font_sizes else 9.0

    # Pre-calculate line text case
    line_texts = defaultdict(list)
    for t in tokens:
        line_texts[(t.get("page", 1), t.get("lineIndex", 0))].append(t)
        
    line_is_all_caps = {}
    for line_key, line_toks in line_texts.items():
        line_str = " ".join(str(tok.get("token", "")) for tok in line_toks).strip()
        line_is_all_caps[line_key] = line_str.isupper() and len(line_str) > 2

    # 1. STYLE CORRELATION ENGINE
    # For a given document pool of tokens, collect all unique style tuple signatures: (font_size, is_bold, is_all_caps)
    unique_styles = set()
    for t in tokens:
        fs = t.get("fontSize", 9.0)
        is_bold = t.get("isBold", False)
        line_key = (t.get("page", 1), t.get("lineIndex", 0))
        is_all_caps = line_is_all_caps.get(line_key, False)
        unique_styles.add((fs, is_bold, is_all_caps))

    # 2. HIERARCHICAL SORTING
    # Sort these unique style tuples in descending order using a composite mathematical importance score:
    # Key formula: (font_size, 1 if is_bold else 0, 1 if is_all_caps else 0)
    # The top-ranked unique style becomes Tier 1, the second becomes Tier 2, the third becomes Tier 3, and all remaining variations drop to General Text.
    sorted_styles = sorted(
        unique_styles,
        key=lambda s: (s[0], 1 if s[1] else 0, 1 if s[2] else 0),
        reverse=True
    )
    
    # Filter for prominence to avoid general text from becoming Tier 1/2/3 when no hierarchy exists
    prominent_styles = []
    for s in sorted_styles:
        fs, is_bold, is_all_caps = s
        is_prominent = (fs > median_font_size) or is_bold or is_all_caps
        if is_prominent:
            prominent_styles.append(s)
            
    tier_map = {}
    if len(prominent_styles) >= 1:
        tier_map[prominent_styles[0]] = 1
    if len(prominent_styles) >= 2:
        tier_map[prominent_styles[1]] = 2
    if len(prominent_styles) >= 3:
        tier_map[prominent_styles[2]] = 3

    # Sort tokens to ensure a clean left-to-right reading order across pages
    tokens = sorted(tokens, key=lambda t: (t.get("page", 1), t.get("lineIndex", 0), t.get("tokenIndex", 0), t.get("x0", 0.0)))

    # Initial grouping by raw line coordinates
    line_map = defaultdict(list)
    for t in tokens:
        key = (t.get("page", 1), t.get("lineIndex", 0))
        line_map[key].append(t)

    sorted_keys = sorted(line_map.keys())
    lines = []
    
    # Phase 1: Handle Horizontal Splits (e.g., separating company names from right-aligned dates)
    for key in sorted_keys:
        line_tokens = sorted(line_map[key], key=lambda t: t.get("x0", 0.0))
        
        # Check if this is a description bullet line
        is_desc_bullet_line = False
        if line_tokens:
            starts_with_bullet = line_tokens[0].get("token", "").strip() in BULLET_CHARS or (len(line_tokens) > 1 and line_tokens[1].get("token", "").strip() in BULLET_CHARS)
            if starts_with_bullet:
                # Find first non-bullet token to determine the line's style tier
                first_non_bullet = None
                for tk in line_tokens:
                    if tk.get("token", "").strip() not in BULLET_CHARS:
                        first_non_bullet = tk
                        break
                
                line_tier = 0
                if first_non_bullet:
                    tk_fs = first_non_bullet.get("fontSize", 9.0)
                    tk_bold = first_non_bullet.get("isBold", False)
                    tk_caps = line_is_all_caps.get((first_non_bullet.get("page", 1), first_non_bullet.get("lineIndex", 0)), False)
                    tk_sig = (tk_fs, tk_bold, tk_caps)
                    line_tier = tier_map.get(tk_sig, 0)
                
                # Classified as description bullet only if calculated style tier is Tier 0 (General Text)
                if line_tier == 0:
                    is_desc_bullet_line = True
                
        current_sub_tokens = []
        for t_idx, t in enumerate(line_tokens):
            if t_idx == 0:
                current_sub_tokens.append(t)
                continue
                
            prev_t = line_tokens[t_idx - 1]
            
            # Identify horizontal gaps (e.g., spaces wider than 35 pixels on the same line index)
            word_gap_skipped = (t.get("x0", 0.0) - prev_t.get("x1", 0.0)) > 35.0 
            
            # Identify bold status shifts or font size shifts (> 0.5 points) - disabled for description bullet lines
            bold_shifted = False
            size_shifted = False
            if not is_desc_bullet_line:
                t_text = t.get("token", "").strip()
                prev_text = prev_t.get("token", "").strip()
                is_bullet_transition = (t_text in BULLET_CHARS) or (prev_text in BULLET_CHARS)
                
                if not is_bullet_transition:
                    bold_shifted = t.get("isBold", False) != prev_t.get("isBold", False)
                    size_shifted = abs(t.get("fontSize", 9.0) - prev_t.get("fontSize", 9.0)) > 0.5
            
            if word_gap_skipped or bold_shifted or size_shifted:
                if current_sub_tokens:
                    lines.append(build_line_record(key[0], key[1], current_sub_tokens))
                current_sub_tokens = [t]
            else:
                current_sub_tokens.append(t)
                
        if current_sub_tokens:
            lines.append(build_line_record(key[0], key[1], current_sub_tokens))

    if not lines:
        return []

    # Phase 2: Adaptive Vertical Paragraph Aggregation
    if not allow_vertical_merge:
        groups = [[line] for line in lines]
    else:
        groups = []
        current_group = []

        for idx, line in enumerate(lines):
            if idx == 0:
                current_group.append(line)
                continue
                
            prev_line = lines[idx - 1]
            should_split = False
            
            # Vertical proximity delta tracking
            vertical_gap = line["y0"] - prev_line["y1"]
            
            # Calculate true layout proximity parameters
            prev_center = (prev_line["x0"] + prev_line["x1"]) / 2.0
            curr_center = (line["x0"] + line["x1"]) / 2.0
            
            is_tight_gap = (-3.0 <= vertical_gap <= 5.5)
            
            # A line can ONLY merge if:
            # 1. The gap is tight
            # 2. Left edges are aligned (delta <= 8px) OR centers are aligned (delta <= 12px)
            can_merge = is_tight_gap and (
                abs(line["content_x0"] - prev_line["content_x0"]) <= 8.0 or 
                abs(curr_center - prev_center) <= 12.0
            )
            
            # Enforce that a line can ONLY merge if can_merge is true. Otherwise, set should_split = True.
            if not can_merge:
                should_split = True
            else:
                # Visual/Contextual boundary checks based strictly on visual/spatial features
                if line["page"] != prev_line["page"]:
                    should_split = True
                elif any(line["text"].startswith(b) for b in BULLET_CHARS):
                    should_split = True
                elif abs(line["fontSize"] - prev_line["fontSize"]) > 1.0:
                    should_split = True
                elif line["isBold"] != prev_line["isBold"]:
                    should_split = True
                else:
                    should_split = False

            if should_split:
                if current_group:
                    groups.append(current_group)
                current_group = [line]
            else:
                current_group.append(line)

        if current_group:
            groups.append(current_group)

    # Phase 3: Final Structuring of Sentence Groups
    sentences_data = []
    has_flat_styles = len(unique_styles) == 1
    last_was_tier1 = False
    for idx, g in enumerate(groups):
        combined_text = " ".join(l["text"] for l in g)
        g_tokens = []
        for l in g:
            g_tokens.extend(l["tokens"])
            
        # Re-verify layout styles for the combined group
        fs, bold = get_dominant_font_info(g_tokens)
        page = g[0]["page"]
        x0 = min(l["x0"] for l in g)
        y0 = min(l["y0"] for l in g)
        x1 = max(l["x1"] for l in g)
        y1 = max(l["y1"] for l in g)

        # Determine Visual Hierarchy Tier for the reconstructed group/row
        g_all_caps = combined_text.isupper() and len(combined_text) > 2
        sig = (fs, bold, g_all_caps)
        tier = tier_map.get(sig, 0)
        
        if has_flat_styles and idx == 0:
            tier = 1
            
        if tier == 1:
            if last_was_tier1:
                tier = 0
            else:
                last_was_tier1 = True
        else:
            last_was_tier1 = False
        
        if tier == 1:
            tier_label = "Tier 1 (Heading/Anchor Peak)"
        elif tier == 2:
            tier_label = "Tier 2 (Sub-Header/Role)"
        elif tier == 3:
            tier_label = "Tier 3 (Sub-Metadata)"
        else:
            tier_label = "General Text"
            
        spatial_array = [
            fs, bold, "plain", page,
            round(y0, 2), round(x0, 2), round(y1, 2), round(x1, 2),
            tier
        ]
        
        sentences_data.append({
            "text": combined_text,
            "spatial": spatial_array,
            "tier_label": tier_label,
            "tokens": g_tokens
        })
        
    return sentences_data
