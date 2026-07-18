import os
import re
import sys
import json
import urllib.parse
from collections import defaultdict
import pandas as pd
import pdfplumber

# Gap larger than this multiple of char size starts a new token (word boundary).
GAP_TOKEN_MULTIPLIER = 0.2
# Tolerance for grouping chars into the same visual line (pdfplumber units).
LINE_TOP_TOLERANCE = 3
# Min length to try fallback split (pipe or camelCase).
MIN_LENGTH_FOR_PIPE_SPLIT = 25
MIN_LENGTH_FOR_CAMEL_SPLIT = 20

def _expand_long_tokens(line_tokens):
    """
    Fallback: split a single (token_text, token_chars) into multiple when the
    token is too long and contains '|' or has camelCase. Yields (text, chars) per part.
    """
    for token_text, token_chars in line_tokens:
        if not token_text or not token_chars:
            continue
        n_chars = len(token_chars)
        if n_chars != len(token_text):
            # Keep as-is if char count doesn't match (safety).
            yield (token_text, token_chars)
            continue
        # Split on pipe when long (e.g. "Hyderabad|7993851335|email@...").
        if len(token_text) >= MIN_LENGTH_FOR_PIPE_SPLIT and "|" in token_text:
            start = 0
            for part in token_text.split("|"):
                part = part.strip()
                if not part:
                    start += 1  # skip delimiter
                    continue
                end = start + len(part)
                if end <= n_chars:
                    sub_chars = token_chars[start:end]
                    yield (part, sub_chars)
                start = end + 1  # skip delimiter
            continue
        # Split on camelCase when long (e.g. "BhagyalaxmiBenkantiwar").
        if len(token_text) >= MIN_LENGTH_FOR_CAMEL_SPLIT and re.search(r"[a-z][A-Z]", token_text):
            positions = [0]
            for m in re.finditer(r"[a-z]([A-Z])", token_text):
                positions.append(m.start(1))
            positions.append(len(token_text))
            for i in range(len(positions) - 1):
                a, b = positions[i], positions[i + 1]
                part = token_text[a:b]
                if part:
                    yield (part, token_chars[a:b])
            continue
        yield (token_text, token_chars)

ABS_GAP_MIN_PT = 1.2

def _chars_to_tokens(line_chars):
    """
    Split line chars into tokens (words) by horizontal gaps and space characters.
    Hyphenated words (e.g. scikit-learn) stay together if no large gap.
    Returns list of (token_text, token_chars).
    """
    if not line_chars:
        return []
    line_chars = sorted(line_chars, key=lambda c: c["x0"])
    tokens = []
    current = []

    for c in line_chars:
        text_val = c.get("text", "")
        # Treat any whitespace character as a token boundary
        if not text_val or text_val.isspace():
            if current:
                t_text = "".join(ch.get("text", "") for ch in current)
                if t_text.strip():
                    tokens.append((t_text.strip(), current))
                current = []
            continue

        if current:
            prev_char = current[-1]
            gap = c["x0"] - prev_char.get("x1", prev_char["x0"])
            size = max(float(c.get("size", 12)), 1)
            # Split if there's a geometric gap
            is_word_boundary = (gap > GAP_TOKEN_MULTIPLIER * size) or (gap >= ABS_GAP_MIN_PT)
            if is_word_boundary:
                t_text = "".join(ch.get("text", "") for ch in current)
                if t_text.strip():
                    tokens.append((t_text.strip(), current))
                current = []
        current.append(c)

    if current:
        t_text = "".join(ch.get("text", "") for ch in current)
        if t_text.strip():
            tokens.append((t_text.strip(), current))
    return tokens

def extract_tokens_from_pdf(file_path):
    doc_id = os.path.basename(file_path).replace(".pdf", "")
    tokens_out = []
    line_index_global = 0

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            chars = page.chars
            hyperlinks = page.hyperlinks
            if not chars:
                continue

            df_chars = pd.DataFrame(chars)
            # Remove completely empty character objects, but keep spaces
            df_chars = df_chars[df_chars["text"] != ""]
            if df_chars.empty:
                continue

            df_chars = df_chars.sort_values(by="top")
            chars_list = df_chars.to_dict("records")

            # Group chars into lines (same as 10mb-tlm pdf_processor)
            grouped_lines = []
            current_line = [chars_list[0]]
            for i in range(1, len(chars_list)):
                if abs(chars_list[i]["top"] - current_line[-1]["top"]) <= LINE_TOP_TOLERANCE:
                    current_line.append(chars_list[i])
                else:
                    grouped_lines.append(current_line)
                    current_line = [chars_list[i]]
            grouped_lines.append(current_line)

            for line_chars in grouped_lines:
                line_chars = sorted(line_chars, key=lambda x: x["x0"])
                line_tokens = _chars_to_tokens(line_chars)
                if not line_tokens:
                    continue
                line_tokens = list(_expand_long_tokens(line_tokens))
                if not line_tokens:
                    continue

                for token_idx, (token_text, token_chars) in enumerate(line_tokens):
                    x0 = min(c["x0"] for c in token_chars)
                    y0 = min(c["top"] for c in token_chars)
                    x1 = max(c["x1"] for c in token_chars)
                    y1 = max(c["bottom"] for c in token_chars)
                    font_sizes = [c.get("size", 12) for c in token_chars]
                    font_size = round(sum(font_sizes) / len(font_sizes), 2) if font_sizes else 12.0
                    font_names = [c.get("fontname", "") for c in token_chars]
                    font_name = max(set(font_names), key=font_names.count).lower() if font_names else ""
                    bold_indicators = ["bold", "black", "heavy", "medium", "cmbx", "cmb", "demi", "semibold", "500", "w6", "w7", "w8", "w9"]
                    is_bold = 1 if any(ind in font_name for ind in bold_indicators) else 0

                    embedded_url = ""
                    max_overlap = 0
                    token_area = (x1 - x0) * (y1 - y0)
                    if token_area > 0:
                        for hl in hyperlinks:
                            hx0, hy0 = hl.get("x0", 0), hl.get("top", 0)
                            hx1, hy1 = hl.get("x1", 0), hl.get("bottom", 0)
                            ix0, iy0 = max(hx0, x0), max(hy0, y0)
                            ix1, iy1 = min(hx1, x1), min(hy1, y1)
                            if ix1 > ix0 and iy1 > iy0:
                                overlap_area = (ix1 - ix0) * (iy1 - iy0)
                                if overlap_area > max_overlap:
                                    max_overlap = overlap_area
                                    uri = hl.get("uri")
                                    embedded_url = urllib.parse.quote(uri, safe="") if uri else ""

                    tokens_out.append({
                        "doc_id": doc_id,
                        "docId": doc_id,
                        "page": page_num + 1,
                        "line_index": line_index_global,
                        "lineIndex": line_index_global,
                        "token_index": token_idx,
                        "tokenIndex": token_idx,
                        "token": token_text,
                        "x0": round(x0, 2),
                        "y0": round(y0, 2),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "font_size": font_size,
                        "fontSize": font_size,
                        "is_bold": is_bold,
                        "isBold": bool(is_bold),
                        "bio_label": "O",
                        "bioLabel": "O",
                        "section": "NONE",
                        "embedded_url": embedded_url,
                        "embeddedUrl": embedded_url,
                    })
                line_index_global += 1

    pipe_split = split_pipe_tokens(tokens_out)
    comma_split = split_comma_tokens(pipe_split)
    date_split = split_combined_date_tokens(comma_split)
    colon_split = split_colon_tokens(date_split)
    bracket_split = split_bracket_tokens(colon_split)
    return clean_tokens(bracket_split)

def split_combined_date_tokens(tokens):
    all_months = 'January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec'
    combined_date_re = re.compile(
        rf"^(?P<sdate>[\(\[\s]*(?:(?:0?[1-9]|1[0-2])[\/\-](?:19|20)\d{{2}}|(?:{all_months})[.,]?[\s\-\/]?['‘’ʼ′´]?\d{{2,4}}|(?:19|20)\d{{2}}))"
        rf"(?P<sep>[\-\u2013\u2014])"
        rf"(?P<edate>(?:Present|Current|Currently|Now|Ongoing|Today|(?:0?[1-9]|1[0-2])[\/\-](?:19|20)\d{{2}}|(?:{all_months})[.,]?[\s\-\/]?['‘’ʼ′´]?\d{{2,4}}|(?:19|20)\d{{2}})[\)\]\s]*)$",
        re.IGNORECASE
    )
    
    expanded = []
    for t in tokens:
        match = combined_date_re.match(t["token"].strip())
        if not match:
            expanded.append(t)
            continue
            
        sdate = match.group("sdate")
        sep = match.group("sep")
        edate = match.group("edate")
        parts = [sdate, sep, edate]
        
        total_chars = sum(len(p) for p in parts)
        total_width = t["x1"] - t["x0"]
        x_cursor = t["x0"]
        
        for idx, part in enumerate(parts):
            part_width = (len(part) / total_chars) * total_width if total_chars > 0 else total_width / len(parts)
            
            # BIO label logic
            bio_label = t.get("bio_label", t.get("bioLabel", "O"))
            if bio_label and bio_label != "O":
                if idx == 0:
                    bio_label = "I-SDATE" if bio_label.startswith("I-") else "B-SDATE"
                elif idx == 1:
                    bio_label = "O"
                else:
                    bio_label = "B-EDATE"
                        
            new_t = t.copy()
            new_t["token"] = part
            new_t["x0"] = round(x_cursor, 2)
            new_t["x1"] = round(x_cursor + part_width, 2)
            new_t["token_index"] = 0
            new_t["tokenIndex"] = 0
            new_t["bio_label"] = bio_label
            new_t["bioLabel"] = bio_label
            expanded.append(new_t)
            x_cursor += part_width
            
    # Re-assign tokenIndex / token_index
    line_counters = {}
    for t in expanded:
        key = f"{t['page']}-{t.get('line_index', t.get('lineIndex', 0))}"
        idx = line_counters.get(key, 0)
        t["token_index"] = idx
        t["tokenIndex"] = idx
        line_counters[key] = idx + 1
        
    return expanded

def split_comma_tokens(tokens):
    expanded = []
    for t in tokens:
        token_str = t.get("token", "")
        if "," not in token_str or token_str == ",":
            expanded.append(t)
            continue
            
        parts = [p for p in re.split(r"(,)", token_str) if p]
        if len(parts) <= 1:
            expanded.append(t)
            continue
            
        total_chars = sum(len(p) for p in parts)
        total_width = t["x1"] - t["x0"]
        x_cursor = t["x0"]
        
        word_idx = 0
        for idx, part in enumerate(parts):
            part_width = (len(part) / total_chars) * total_width if total_chars > 0 else total_width / len(parts)
            
            bio_label = t.get("bio_label", t.get("bioLabel", "O"))
            if bio_label and bio_label != "O":
                if part == ",":
                    bio_label = "O"
                else:
                    base = re.sub(r"^[BI]-", "", bio_label)
                    if word_idx == 0:
                        bio_label = f"I-{base}" if bio_label.startswith("I-") else f"B-{base}"
                    else:
                        bio_label = f"I-{base}"
                    word_idx += 1
                    
            new_t = t.copy()
            new_t["token"] = part
            new_t["x0"] = round(x_cursor, 2)
            new_t["x1"] = round(x_cursor + part_width, 2)
            new_t["token_index"] = 0
            new_t["tokenIndex"] = 0
            new_t["bio_label"] = bio_label
            new_t["bioLabel"] = bio_label
            expanded.append(new_t)
            x_cursor += part_width
            
    # Re-assign tokenIndex / token_index
    line_counters = {}
    for t in expanded:
        key = f"{t['page']}-{t.get('line_index', t.get('lineIndex', 0))}"
        idx = line_counters.get(key, 0)
        t["token_index"] = idx
        t["tokenIndex"] = idx
        line_counters[key] = idx + 1
        
    return expanded

def split_colon_tokens(tokens):
    expanded = []
    for t in tokens:
        token_str = t.get("token", "")
        if ":" not in token_str or token_str == ":":
            expanded.append(t)
            continue
            
        parts = [p for p in re.split(r"(:)", token_str) if p]
        if len(parts) <= 1:
            expanded.append(t)
            continue
            
        total_chars = sum(len(p) for p in parts)
        total_width = t["x1"] - t["x0"]
        x_cursor = t["x0"]
        
        for idx, part in enumerate(parts):
            part_width = (len(part) / total_chars) * total_width if total_chars > 0 else total_width / len(parts)
            
            bio_label = t.get("bio_label", t.get("bioLabel", "O"))
            if bio_label and bio_label != "O":
                if idx < len(parts) - 1:
                    bio_label = "O"
                else:
                    bio_label = bio_label
                    
            new_t = t.copy()
            new_t["token"] = part
            new_t["x0"] = round(x_cursor, 2)
            new_t["x1"] = round(x_cursor + part_width, 2)
            new_t["token_index"] = 0
            new_t["tokenIndex"] = 0
            new_t["bio_label"] = bio_label
            new_t["bioLabel"] = bio_label
            expanded.append(new_t)
            x_cursor += part_width
            
    # Re-assign tokenIndex / token_index
    line_counters = {}
    for t in expanded:
        key = f"{t['page']}-{t.get('line_index', t.get('lineIndex', 0))}"
        idx = line_counters.get(key, 0)
        t["token_index"] = idx
        t["tokenIndex"] = idx
        line_counters[key] = idx + 1
        
    return expanded

def extract_tokens(pdf_path):
    tokens = extract_tokens_from_pdf(pdf_path)
    doc_id = os.path.basename(pdf_path).replace(".pdf", "")
    return {
        "resumeId": doc_id,
        "tokens": tokens
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_tokens.py <path_to_pdf> <path_to_output_json>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    out_json = sys.argv[2]
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF path not found: {pdf_path}")
        sys.exit(1)
        
    print(f"Extracting tokens from: {pdf_path}...")
    res = extract_tokens(pdf_path)
    
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        
    print(f"Extraction successful! Saved {len(res['tokens'])} tokens to {out_json}")

def split_pipe_tokens(tokens):
    expanded = []
    for t in tokens:
        token_str = t.get("token", "")
        if "|" not in token_str:
            expanded.append(t)
            continue
        parts = [p.strip() for p in token_str.split("|") if p.strip()]
        if len(parts) <= 1:
            expanded.append(t)
            continue
        total_chars = sum(len(p) for p in parts)
        total_width = t["x1"] - t["x0"]
        x_cursor = t["x0"]
        for idx, part in enumerate(parts):
            part_width = (len(part) / total_chars) * total_width if total_chars > 0 else total_width / len(parts)
            new_t = t.copy()
            new_t["token"] = part
            new_t["x0"] = round(x_cursor, 2)
            new_t["x1"] = round(x_cursor + part_width, 2)
            new_t["token_index"] = 0
            new_t["tokenIndex"] = 0
            expanded.append(new_t)
            x_cursor += part_width
            
    # Re-assign tokenIndex / token_index
    line_counters = {}
    for t in expanded:
        key = f"{t['page']}-{t.get('line_index', t.get('lineIndex', 0))}"
        idx = line_counters.get(key, 0)
        t["token_index"] = idx
        t["tokenIndex"] = idx
        line_counters[key] = idx + 1
        
    return expanded

_BRACKET_RE = re.compile(r"([()\[\]{}])")

def tokenize_brackets(raw: str) -> list[str]:
    pieces = _BRACKET_RE.split(raw)
    return [p for p in pieces if p]

def needs_bracket_split(token: str) -> bool:
    t = token.strip()
    if "://" in t or t.startswith("//") or t.startswith("www.") or "@" in t:
        return False
    has_bracket = any(c in t for c in "()[]{}")
    if not has_bracket:
        return False
    has_other = any(c not in "()[]{}" for c in t)
    if not has_other:
        return False
    return True

def split_bracket_tokens(tokens):
    expanded = []
    line_shift = {}
    for t in tokens:
        page = t.get("page")
        line = t.get("line_index", t.get("lineIndex", 0))
        key = (page, line)
        
        shift = line_shift.get(key, 0)
        t_copy = t.copy()
        t_copy["tokenIndex"] = t.get("tokenIndex", 0) + shift
        t_copy["token_index"] = t.get("token_index", 0) + shift
        
        raw_str = t.get("token", "")
        if needs_bracket_split(raw_str):
            pieces = tokenize_brackets(raw_str)
            total_chars = sum(len(p) for p in pieces)
            x0_base = t["x0"]
            total_width = t["x1"] - x0_base
            char_width = total_width / total_chars if total_chars > 0 else 0.0
            
            parent_label = t.get("bioLabel", t.get("bio_label", "O"))
            
            labels = []
            has_seen_first_word = False
            for piece in pieces:
                if any(c in piece for c in "()[]{}"):
                    labels.append("O")
                else:
                    if parent_label == "O":
                        labels.append("O")
                    elif parent_label.startswith("B-"):
                        entity_type = parent_label[2:]
                        if not has_seen_first_word:
                            labels.append(parent_label)
                            has_seen_first_word = True
                        else:
                            labels.append(f"I-{entity_type}")
                    elif parent_label.startswith("I-"):
                        labels.append(parent_label)
                    else:
                        labels.append(parent_label)
            
            sub = []
            cursor_x = x0_base
            for piece, label in zip(pieces, labels):
                pw = len(piece) * char_width
                st = t_copy.copy()
                st["token"] = piece
                st["bioLabel"] = label
                st["bio_label"] = label
                st["x0"] = round(cursor_x, 2)
                st["x1"] = round(cursor_x + pw, 2)
                sub.append(st)
                cursor_x += pw
                
            for j, st in enumerate(sub):
                st["tokenIndex"] = t_copy["tokenIndex"] + j
                st["token_index"] = t_copy["tokenIndex"] + j
                
            extra = len(sub) - 1
            line_shift[key] = line_shift.get(key, 0) + extra
            expanded.extend(sub)
        else:
            expanded.append(t_copy)
            
    # Re-assign tokenIndex / token_index
    line_counters = {}
    for t in expanded:
        key = f"{t['page']}-{t.get('line_index', t.get('lineIndex', 0))}"
        idx = line_counters.get(key, 0)
        t["token_index"] = idx
        t["tokenIndex"] = idx
        line_counters[key] = idx + 1
        
    return expanded

def clean_tokens(tokens):
    if not tokens:
        return []

    line_map = defaultdict(list)
    for t in tokens:
        key = (t.get("page", 0), t.get("lineIndex", t.get("line_index", 0)))
        line_map[key].append(t)
        
    valid_line_keys = set()
    for key, line_tokens in line_map.items():
        text = " ".join(t.get("token", "") for t in line_tokens)
        if re.search(r'[a-zA-Z0-9]', text):
            valid_line_keys.add(key)
            
    filtered_tokens = [t for t in tokens if (t.get("page", 0), t.get("lineIndex", t.get("line_index", 0))) in valid_line_keys]
    
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
            
    final_tokens = []
    for t in filtered_tokens:
        tok_text = t.get("token", "")
        if tok_text is not None and tok_text.strip() != "":
            final_tokens.append(t)
            
    return final_tokens

if __name__ == "__main__":
    main()
