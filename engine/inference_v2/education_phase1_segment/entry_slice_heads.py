"""Resolve education entry head lines from step-5 boundaries or MongoDB y0 mapping."""

from __future__ import annotations

from inference_v2.overlay_mongo_labels import load_mongo_entry_heads


def resolve_education_boundary_heads(tokens: list[dict]) -> set[tuple[int, int]]:
    """Lines with B-EDU_START from education_phase2_divider (step 5)."""
    heads: set[tuple[int, int]] = set()
    for t in tokens:
        label = t.get("bioLabel") or t.get("bio_label") or "O"
        if label == "B-EDU_START":
            heads.add((
                int(t.get("page", 0)),
                int(t.get("lineIndex", t.get("line_index", 0))),
            ))
    return heads


def resolve_education_entry_heads(resume_id: str, filtered_tokens: list[dict]) -> set[tuple[int, int]]:
    """Convert mongo (page, lineIndex) heads to inference (page, lineIndex) using y0 alignment."""
    mongo_heads = load_mongo_entry_heads(resume_id, "EDUCATION")
    if not mongo_heads or not filtered_tokens:
        return set()

    try:
        from pymongo import MongoClient
        from core.config import MONGO_DB, MONGO_URI
    except ImportError:
        return set()

    client = MongoClient(MONGO_URI)
    doc = client[MONGO_DB].resumes.find_one({"resumeId": resume_id})
    client.close()
    if not doc:
        return set()

    head_bands: list[tuple[int, float]] = []
    for page, line in sorted(mongo_heads):
        line_toks = [
            t for t in doc.get("tokens", [])
            if t.get("page") == page and t.get("lineIndex") == line
        ]
        if line_toks:
            y0 = min(float(t.get("y0", 0.0)) for t in line_toks)
            head_bands.append((page, y0))

    if not head_bands:
        return set()

    resolved: set[tuple[int, int]] = set()
    for page, head_y0 in head_bands:
        for t in filtered_tokens:
            if t.get("page") != page:
                continue
            if abs(float(t.get("y0", 0.0)) - head_y0) > 3.0:
                continue
            resolved.add((int(t["page"]), int(t["lineIndex"])))
            break

    return resolved
