import re
import random
from collections import defaultdict

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

def sort_tokens_by_reading_order(tokens: list[dict]) -> list[dict]:
    """Untangle interleaved tokens from multi-column layouts at runtime.
    Groups tokens by page and section, detects vertical gutters within each section,
    and sorts two-column sections column-by-column, while preserving single-column
    sections row-by-row.
    """
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
        sorted_groups = sorted(page_groups[page], key=lambda x: x[1])
        
        for (page, section), _, pts in sorted_groups:
            # Skip two-column sorting for PERSONAL, SUMMARY, and SKILLS
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
            
            has_gutter = False
            if best_x is not None and (min_overlaps <= 3 or min_overlaps < 0.08 * len(pts)):
                line_tokens = defaultdict(list)
                for t in pts:
                    line_tokens[t["lineIndex"]].append(t)
                    
                crossing_lines = 0
                middle_gaps = 0
                for line_idx, lts in line_tokens.items():
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

def clean_non_text_tokens(tokens: list[dict]) -> list[dict]:
    """Filter out tokens belonging to lines that have no alphanumeric text."""
    line_map = defaultdict(list)
    for t in tokens:
        key = (t.get("page"), t.get("lineIndex"))
        line_map[key].append(t)
        
    valid_line_keys = set()
    for key, line_tokens in line_map.items():
        text = " ".join(t.get("token", "") for t in line_tokens)
        if re.search(r'[a-zA-Z0-9]', text):
            valid_line_keys.add(key)
            
    filtered = [t for t in tokens if (t.get("page"), t.get("lineIndex")) in valid_line_keys]
    
    # Strip cid:NNN sequences
    i = 0
    while i < len(filtered) - 2:
        t0 = filtered[i]
        t1 = filtered[i+1]
        t2 = filtered[i+2]
        if t0.get("token") == "cid" and t1.get("token") == ":" and t2.get("token", "").isdigit():
            t0["token"] = ""
            t1["token"] = ""
            t2["token"] = ""
            i += 3
        else:
            i += 1
            
    return filtered

def _normalize_spatial(tokens: list[dict], all_tokens: list[dict] | None = None, augment: bool = False) -> list[list[float]]:
    """Generate 20D spatial features per token."""
    INLINE_DELIMITERS = {"-", "–", "—", "|", ",", "•", "●", "❖", "▪", ":", "~", "/", "(", ")"}
    
    current_line_idx = -1
    distance_from_delimiter = 0
    delimiter_dists = []
    
    for idx in range(len(tokens)):
        curr_tok = tokens[idx]
        tok_str = curr_tok.get("token", "").strip()
        line_idx = curr_tok.get("lineIndex", 0)
        
        if line_idx != current_line_idx:
            current_line_idx = line_idx
            distance_from_delimiter = 0
        elif tok_str in INLINE_DELIMITERS:
            distance_from_delimiter = 0
        else:
            distance_from_delimiter += 1
            
        norm_dist = min(float(distance_from_delimiter) / 10.0, 1.0)
        delimiter_dists.append(norm_dist)

    deltas = []
    for idx in range(len(tokens)):
        if idx == 0:
            line_delta = 0.0
            font_delta = 0.0
            bold_delta = 0.0
        else:
            prev_tok = tokens[idx - 1]
            curr_tok = tokens[idx]
            
            line_delta = float(curr_tok.get("lineIndex", 0) - prev_tok.get("lineIndex", 0))
            font_delta = float(curr_tok.get("fontSize", 0.0) - prev_tok.get("fontSize", 0.0))
            bold_delta = float(int(curr_tok.get("isBold", False)) - int(prev_tok.get("isBold", False)))
            
            if curr_tok.get("page", 0) != prev_tok.get("page", 0):
                line_delta = 1.0
        deltas.append((line_delta, font_delta, bold_delta))

    ref_tokens = all_tokens if all_tokens is not None else tokens
    
    page_max_lines = defaultdict(int)
    for t in ref_tokens:
        page_max_lines[t.get("page", 0)] = max(page_max_lines[t.get("page", 0)], t.get("lineIndex", 0))

    pages = defaultdict(list)
    for i, t in enumerate(tokens):
        pages[t.get("page", 0)].append((i, t))

    result = [None] * len(tokens)
    for items in pages.values():
        xs  = [t.get("x0", 0.0)       for _, t in items]
        x1s = [t.get("x1", 0.0)       for _, t in items]
        ys  = [t.get("y0", 0.0)       for _, t in items]
        y1s = [t.get("y1", 0.0)       for _, t in items]
        fs  = [t.get("fontSize", 9.0) for _, t in items]

        if not xs or not x1s or not ys or not y1s:
            continue

        xmin, xmax = min(xs),  max(x1s)
        ymin, ymax = min(ys),  max(y1s)
        pw = max(xmax - xmin, 1e-6)
        ph = max(ymax - ymin, 1e-6)
        fmax = max(fs) or 1.0

        items_sorted = sorted(items, key=lambda x: x[0])
        prev_y1 = None
        prev_token = None
        for idx, t in items_sorted:
            x0n  = (t.get("x0", 0.0) - xmin) / pw
            y0n  = (t.get("y0", 0.0) - ymin) / ph
            wn   = (t.get("x1", 0.0) - t.get("x0", 0.0)) / pw
            hn   = (t.get("y1", 0.0) - t.get("y0", 0.0)) / ph
            
            if prev_token is not None and t.get("page", 0) == prev_token.get("page", 0) and t.get("lineIndex", 0) == prev_token.get("lineIndex", 0):
                dx_n = t.get("x0", 0.0) - prev_token.get("x1", 0.0)
                is_line_start = 0.0
            else:
                dx_n = 0.0
                is_line_start = 1.0

            if augment:
                noise_x0 = random.gauss(0, 0.015)
                noise_x1 = random.gauss(0, 0.015)
                x0n_aug = max(min(x0n + noise_x0, 1.0), 0.0)
                x1n_aug = max(min((x0n + wn) + noise_x1, 1.0), 0.0)
                if x1n_aug < x0n_aug:
                    x1n_aug = x0n_aug
                x0n = x0n_aug
                wn = x1n_aug - x0n_aug
                
                noise_y0 = random.gauss(0, 0.005)
                noise_y1 = random.gauss(0, 0.005)
                y0n_aug = max(min(y0n + noise_y0, 1.0), 0.0)
                y1n_aug = max(min((y0n + hn) + noise_y1, 1.0), 0.0)
                if y1n_aug < y0n_aug:
                    y1n_aug = y0n_aug
                y0n = y0n_aug
                hn = y1n_aug - y0n_aug
                
            bold = float(t.get("isBold", False))
            text = t.get("token", "")
            caps = float(len(text) > 1 and text.isupper())
            fn   = t.get("fontSize", 9.0) / fmax
            abs_y = t.get("y0", 0.0) / max(max(y1s), 1.0)
            
            dy_n = 0.0 if prev_y1 is None else (t.get("y0", 0.0) - prev_y1) / ph
            
            is_after_inline_delimiter = 0.0
            inline_seps = {"|", "-", "—", "\\", "/", "(", ")"}
            if idx > 0 and tokens[idx - 1].get("token", "").strip() in inline_seps:
                is_after_inline_delimiter = 1.0
            elif idx > 1 and tokens[idx - 2].get("token", "").strip() in inline_seps:
                is_after_inline_delimiter = 1.0

            deg_pat = re.compile(r'\b(b\.?tech|m\.?c\.?a|b\.?c\.?a|m\.?b\.?a|bachelor|master|diploma|cbse|icse)\b', re.IGNORECASE)
            is_degree_anchor = 1.0 if deg_pat.search(text) else 0.0

            inst_pat = re.compile(r'\b(university|college|instit|school|academy)\b', re.IGNORECASE)
            is_institution_anchor = 1.0 if inst_pat.search(text) else 0.0
            
            connectives = {"and", "of", "in", "with", "from", "to"}
            is_connective_word = 1.0 if text.strip().lower() in connectives else 0.0
            
            result[idx] = [x0n, y0n, wn, hn, bold, caps, fn, abs_y, dx_n, dy_n, is_line_start, 0.0] + list(deltas[idx]) + [delimiter_dists[idx]] + [is_after_inline_delimiter, is_degree_anchor, is_institution_anchor, is_connective_word]
            prev_y1 = t.get("y1", 0.0)
            prev_token = t

    return [r if r is not None else [0.0] * 20 for r in result]
