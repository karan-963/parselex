import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple

def normalize_text(text: str) -> str:
    """Removes all non-alphabetic characters and lowercases the text."""
    if not text:
        return ""
    return re.sub(r'[^a-z]', '', text.lower())

def median_font_size_of(tokens: List[Dict[str, Any]]) -> float:
    """Calculates median font size of tokens."""
    if not tokens:
        return 0.0
    sizes = sorted([float(t.get("fontSize", 9.0)) for t in tokens])
    mid = len(sizes) // 2
    if len(sizes) % 2 == 0:
        return (sizes[mid - 1] + sizes[mid]) / 2.0
    return sizes[mid]

def median_inter_token_gap(line_map: Dict[Tuple[int, int], List[Dict[str, Any]]]) -> float:
    """Calculates median gap between horizontal tokens on same lines."""
    gaps = []
    for line_tokens in line_map.values():
        sorted_tokens = sorted(line_tokens, key=lambda t: t.get("x0", 0.0))
        for i in range(len(sorted_tokens) - 1):
            gap = sorted_tokens[i + 1].get("x0", 0.0) - sorted_tokens[i].get("x1", 0.0)
            if gap > 0:
                gaps.append(gap)
    if not gaps:
        return 10.0
    gaps.sort()
    mid = len(gaps) // 2
    if len(gaps) % 2 == 0:
        return (gaps[mid - 1] + gaps[mid]) / 2.0
    return gaps[mid]

def median_line_height_of(tokens: List[Dict[str, Any]]) -> float:
    """Calculates median vertical gap between lines on the same page."""
    gaps = []
    for i in range(1, len(tokens)):
        prev = tokens[i - 1]
        curr = tokens[i]
        if prev.get("page") == curr.get("page") and prev.get("lineIndex") != curr.get("lineIndex"):
            gap = curr.get("y0", 0.0) - prev.get("y1", 0.0)
            if gap > 0:
                gaps.append(gap)
    if not gaps:
        token_heights = [t.get("y1", 0.0) - t.get("y0", 0.0) for t in tokens]
        token_heights = [h for h in token_heights if h > 0]
        if not token_heights:
            return 12.0
        token_heights.sort()
        return token_heights[len(token_heights) // 2]
    gaps.sort()
    return gaps[len(gaps) // 2]

def sort_tokens_by_reading_order(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Untangle interleaved tokens from multi-column layouts."""
    if not tokens:
        return []
    # Entity sections are single-column job/edu/project blocks. Bullet + wrapped
    # continuation on the same PDF line falsely triggers gutter detection.
    _ROW_ORDER_SECTIONS = {"EXPERIENCE", "EDUCATION", "PROJECT"}
    page_section_tokens = defaultdict(list)
    for t in tokens:
        page_section_tokens[(t["page"], t.get("section", "UNKNOWN"))].append(t)
    sorted_all_tokens = []
    page_groups = defaultdict(list)
    for (page, section), group_tokens in page_section_tokens.items():
        avg_y0 = sum(t["y0"] for t in group_tokens) / len(group_tokens)
        page_groups[page].append(((page, section), avg_y0, group_tokens))
    for page in sorted(page_groups.keys()):
        sorted_groups = sorted(page_groups[page], key=lambda x: x[1])
        for (page, section), _, pts in sorted_groups:
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
                for row in rows:
                    row.sort(key=lambda t: t["x0"])
                    sorted_section.extend([dict(t) for t in row])
                sorted_all_tokens.extend(sorted_section)
                continue
            if section in _ROW_ORDER_SECTIONS:
                sorted_section = sorted(pts, key=lambda t: (t["lineIndex"], t["x0"]))
                sorted_all_tokens.extend([dict(t) for t in sorted_section])
                continue
            xs = [t["x0"] for t in pts]
            x1s = [t["x1"] for t in pts]
            if not xs or not x1s:
                sorted_all_tokens.extend(pts)
                continue
            xmin, xmax = min(xs), max(x1s)
            width = xmax - xmin
            scan_start = int(xmin + 0.05 * width)
            scan_end = int(xmin + 0.55 * width)
            best_x = None
            min_overlaps = float("inf")
            if scan_end > scan_start:
                import re
                for x in range(scan_start, scan_end + 1):
                    overlaps = sum(
                        1 for t in pts
                        if t["x0"] < x < t["x1"] and bool(re.search(r"[a-zA-Z0-9]", t.get("token", "")))
                    )
                    if overlaps < min_overlaps:
                        min_overlaps = overlaps
                        best_x = x
            has_gutter = False
            if best_x is not None and (min_overlaps <= 3 or min_overlaps < 0.08 * len(pts)):
                line_tokens = defaultdict(list)
                for t in pts:
                    line_tokens[t["lineIndex"]].append(t)
                middle_gaps = 0
                for line_idx, lts in line_tokens.items():
                    lts_sorted = sorted(lts, key=lambda t: t["x0"])
                    if any(t["x0"] < best_x < t["x1"] for t in lts_sorted):
                        continue
                    t_left = max((t for t in lts_sorted if t["x1"] <= best_x), key=lambda t: t["x1"], default=None)
                    t_right = min((t for t in lts_sorted if t["x0"] >= best_x), key=lambda t: t["x0"], default=None)
                    if t_left is not None and t_right is not None:
                        if 10 <= (t_right["x0"] - t_left["x1"]) <= 150:
                            middle_gaps += 1
                
                if middle_gaps >= 2:
                    has_gutter = True
            if has_gutter:
                g_mid = best_x
                left = sorted([t for t in pts if (t["x0"] + t["x1"]) / 2 < g_mid], key=lambda t: (t["y0"], t["x0"]))
                right = sorted([t for t in pts if (t["x0"] + t["x1"]) / 2 >= g_mid], key=lambda t: (t["y0"], t["x0"]))
                sorted_section = left + right
            else:
                sorted_section = sorted(pts, key=lambda t: (t["lineIndex"], t["x0"]))
            sorted_all_tokens.extend(sorted_section)
    page_line_counters = defaultdict(int)
    page_prev_tokens = {}
    page_prev_orig_line_idx = {}
    for t in sorted_all_tokens:
        page = t.get("page", 0)
        prev_t = page_prev_tokens.get(page)
        orig_li = t.get("lineIndex", 0)
        if prev_t is None:
            t["lineIndex"] = 0
        else:
            is_same_line = True
            if t.get("page") != prev_t.get("page"):
                is_same_line = False
            elif orig_li != page_prev_orig_line_idx.get(page):
                is_same_line = False
            elif t.get("section") != prev_t.get("section"):
                is_same_line = False
            else:
                gap = t.get("x0", 0.0) - prev_t.get("x1", 0.0)
                if gap > 30.0:
                    is_same_line = False
            if not is_same_line:
                page_line_counters[page] += 1
            t["lineIndex"] = page_line_counters[page]
        page_prev_tokens[page] = t
        page_prev_orig_line_idx[page] = orig_li
    return sorted_all_tokens
