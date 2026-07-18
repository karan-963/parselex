"""Helpers for education phase-2 boundary evaluation reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any


def collect_education_line_coords(segments: list) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for seg in segments:
        from dataset import is_education_segment

        if not is_education_segment(seg):
            continue
        for tok in seg.get("tokens", []):
            if tok and "page" in tok and "lineIndex" in tok:
                coords.add((tok["page"], tok["lineIndex"]))
    return coords


def parse_entry_heads(doc: dict, tokens: list[dict] | None = None) -> set[tuple[int, int]]:
    heads = [
        h for h in (doc.get("educationEntryHeads") or [])
        if isinstance(h, dict) and "page" in h and "lineIndex" in h
    ]
    if tokens is not None:
        from education_entry_heads import resolve_education_entry_heads
        return resolve_education_entry_heads(tokens, heads)
    return {(int(h["page"]), int(h["lineIndex"])) for h in heads}


def scope_heads_to_education(
    raw_heads: set[tuple[int, int]],
    education_lines: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    in_scope = raw_heads & education_lines
    orphan = raw_heads - education_lines
    return in_scope, orphan


def format_pl_coord(page: int, line: int) -> str:
    return f"P{page}-L{line}"


def format_pl_coords(coords: set[tuple[int, int]]) -> str:
    if not coords:
        return "—"
    return ", ".join(format_pl_coord(p, l) for p, l in sorted(coords))


def lines_for_role(coords: set[tuple[int, int]], line_set: set[tuple[int, int]]) -> str:
    return format_pl_coords({c for c in coords if c in line_set})


def line_text_lookup(segments: list) -> dict[tuple[int, int], str]:
    parts: dict[tuple[int, int], list[str]] = defaultdict(list)
    for seg in segments:
        for tok in seg.get("tokens", []):
            if not tok:
                continue
            key = (tok.get("page"), tok.get("lineIndex"))
            text = (tok.get("token") or tok.get("text") or "").strip()
            if text:
                parts[key].append(text)
    return {k: " ".join(v) for k, v in parts.items()}


def sort_resume_entries(entries: list[dict]) -> list[dict]:
    """Worst FBA first; tie-break by total boundary errors (FP+FN)."""
    return sorted(entries, key=lambda e: (e["fba"], -(e["fp"] + e["fn"])))


def _nearest_pred_line(
    gt_coord: tuple[int, int],
    pred_lines: set[tuple[int, int]],
) -> tuple[int, int] | None:
    if not pred_lines:
        return None
    page, line = gt_coord
    same_page = sorted((p, l) for p, l in pred_lines if p == page)
    if not same_page:
        return min(pred_lines, key=lambda c: (abs(c[0] - page), abs(c[1] - line)))
    return min(same_page, key=lambda c: abs(c[1] - line))


def build_boundary_comparison_rows(
    gt_line_set: set[tuple[int, int]],
    pred_line_set: set[tuple[int, int]],
    physical_lines: dict[tuple[int, int], str],
    education_lines: set[tuple[int, int]],
) -> list[dict]:
    """One row per scored GT or predicted head with separate GT/Pred line text."""
    rows: list[dict] = []
    scored = sorted(gt_line_set | pred_line_set)
    for coord in scored:
        if coord not in education_lines:
            continue
        page, line = coord
        gt_hit = coord in gt_line_set
        pred_hit = coord in pred_line_set
        line_text = physical_lines.get(coord, "").strip()

        if gt_hit and pred_hit:
            status = "✓ MATCH"
            pred_coord = coord
            pred_text = line_text
        elif gt_hit:
            status = "FN"
            pred_coord = _nearest_pred_line(coord, pred_line_set)
            pred_text = physical_lines.get(pred_coord, "").strip() if pred_coord else ""
        elif pred_hit:
            status = "FP"
            pred_coord = coord
            pred_text = line_text
        else:
            continue

        rows.append({
            "status": status,
            "page": page,
            "line": line,
            "pl": format_pl_coord(page, line),
            "gt": "[START]" if gt_hit else "—",
            "pred": "[START]" if pred_hit else "—",
            "line_text": line_text or "_(empty)_",
            "gt_text": line_text if gt_hit else "—",
            "pred_pl": format_pl_coord(*pred_coord) if pred_coord and not pred_hit else (
                format_pl_coord(page, line) if pred_hit else "—"
            ),
            "pred_text": pred_text if pred_text else "—",
        })
    return rows


def comparison_row_md(row: dict) -> str:
    gt_text = (row["gt_text"] or "—").replace("|", "\\|")
    pred_text = (row["pred_text"] or "—").replace("|", "\\|")
    line_text = (row["line_text"] or "—").replace("|", "\\|")
    return (
        f"| {row['status']} | `{row['pl']}` | {row['gt']} | {row['pred']} | "
        f"{gt_text} | `{row['pred_pl']}` | {pred_text} | {line_text} |\n"
    )


COMPARISON_HEADER = (
    "| Status | GT Line | GT | Pred | GT Text | Pred Line | Pred Text | Line @ coord |\n"
    "|---|---|:---:|:---:|---|---:|---|---|\n"
)


def diagnostic_row_md(
    physical_line: str,
    gt_sym: str,
    pred_sym: str,
    is_match: bool,
    text: str,
    gt_text: str = "",
    pred_text: str = "",
    pred_line: str = "",
) -> str:
    if is_match:
        status = "✓"
    elif gt_sym == "[START]" and pred_sym in (".", "—"):
        status = "FN"
    elif pred_sym == "[START]" and gt_sym in (".", "—"):
        status = "FP"
    else:
        status = "·"
    gt_col = (gt_text or text or "—").replace("|", "\\|")
    pred_col = (pred_text or "—").replace("|", "\\|")
    pred_pl = pred_line or "—"
    return (
        f"| `{physical_line}` | `{gt_sym}` | `{pred_sym}` | {status} | "
        f"{gt_col} | `{pred_pl}` | {pred_col} |\n"
    )


DIAGNOSTIC_HEADER = (
    "| GT Line | GT | Pred | Status | GT Text | Pred Line | Pred Text |\n"
    "|---|:---:|:---:|:---:|---|---:|---|\n"
)


def write_per_resume_md(
    path: str,
    resume_id: str,
    entry: dict,
    orphan_details: list[tuple[str, str]],
) -> None:
    with open(path, "w") as mdf:
        mdf.write(f"# Education Boundary Diagnostic: {resume_id}\n\n")
        mdf.write("| Metric | Value |\n|---|---|\n")
        mdf.write(f"| **Boundary FBA** | {entry['fba']:.2f}% |\n")
        mdf.write(f"| **GT Education Heads (in EDUCATION section)** | {entry['gt_heads']} |\n")
        if entry.get("orphan_gt"):
            mdf.write(f"| **Orphan GT Heads (outside EDUCATION)** | {entry['orphan_gt']} |\n")
        mdf.write(f"| **True Positives (TP)** | {entry['tp']} |\n")
        mdf.write(f"| **False Positives (FP)** | {entry['fp']} |\n")
        mdf.write(f"| **False Negatives (FN)** | {entry['fn']} |\n\n")

        mdf.write(
            "_`—` in GT/Pred columns = no boundary at that coordinate. "
            "FN rows show the nearest predicted line in **Pred Line** / **Pred Text**._\n\n"
        )

        comparison = entry.get("line_comparison", [])
        if comparison:
            mdf.write("## Entry line comparison\n\n")
            mdf.write(COMPARISON_HEADER)
            for row in comparison:
                mdf.write(comparison_row_md(row))
            mdf.write("\n")

        if orphan_details:
            mdf.write("## Orphan GT Heads (labeled but no EDUCATION section tokens)\n\n")
            mdf.write("| GT Line | Line Text |\n|---|---|\n")
            for coord, text in orphan_details:
                mdf.write(f"| `{coord}` | {text or '_(no token text)_'} |\n")
            mdf.write("\n")

        mdf.write("## Perfect Matches (Boundary TPs)\n\n")
        if entry["perfect_matches"]:
            mdf.write(DIAGNOSTIC_HEADER)
            for row in entry["perfect_matches"]:
                mdf.write(diagnostic_row_md(*row))
        else:
            mdf.write("No perfect boundary matches (TP = 0).\n")
        mdf.write("\n")

        fn_rows = entry.get("false_negatives", [])
        fp_rows = entry.get("false_positives", [])
        oow_rows = entry.get("fn_out_of_window", [])

        mdf.write("## False Negatives (GT head missed)\n\n")
        mdf.write(
            "_Status **FN** = GT marks this physical line as an entry head, model does not. "
            "Matching is by **(page, lineIndex)** coordinate._\n\n"
        )
        if fn_rows or oow_rows:
            mdf.write(DIAGNOSTIC_HEADER)
            for row in fn_rows:
                mdf.write(diagnostic_row_md(*row))
            for row in oow_rows:
                mdf.write(diagnostic_row_md(*row))
        else:
            mdf.write("No false negatives.\n")
        mdf.write("\n")

        mdf.write("## False Positives (predicted head, no GT)\n\n")
        if fp_rows:
            mdf.write(DIAGNOSTIC_HEADER)
            for row in fp_rows:
                mdf.write(diagnostic_row_md(*row))
        else:
            mdf.write("No false positives.\n")
