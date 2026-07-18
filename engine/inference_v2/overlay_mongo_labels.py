"""Overlay MongoDB field bioLabels onto extracted tokens for segmentation parity."""

from __future__ import annotations

import re


def _coord_key(t: dict) -> tuple:
    return (t.get("page"), round(float(t.get("x0", 0)), 2), round(float(t.get("y0", 0)), 2))


def _is_substring_token_match(inf_tok: str, db_tok: str) -> bool:
    if inf_tok == db_tok:
        return True
    if inf_tok not in db_tok:
        return False
    idx = db_tok.index(inf_tok)
    before = "" if idx == 0 else db_tok[idx - 1]
    after = "" if idx + len(inf_tok) >= len(db_tok) else db_tok[idx + len(inf_tok)]
    ok_before = not before or not re.search(r"[a-zA-Z0-9]", before)
    ok_after = not after or not re.search(r"[a-zA-Z0-9]", after)
    return ok_before and ok_after


_GT_DISABLED_SENTINEL = "__gt_disabled__"


def resolve_mongo_resume_id(resume_id: str, slug: str | None = None) -> str:
    """Map inference doc hash / slug to MongoDB resumeId when possible.

    Returns a non-matching sentinel for non-default resumes so ground truth is
    never read for uploaded resumes (GT is reserved for the demo reference).
    """
    from inference_v2.gt_gate import is_gt_enabled

    if not is_gt_enabled(resume_id, slug):
        return _GT_DISABLED_SENTINEL

    try:
        from pymongo import MongoClient
        from core.config import MONGO_DB, MONGO_URI
        from inference_v2.storage import sanitize_basename
    except ImportError:
        return resume_id

    candidates: list[str] = []

    def add(value: str | None) -> None:
        if value and value not in candidates:
            candidates.append(value)

    add(resume_id)
    add(sanitize_basename(resume_id))
    if slug:
        add(sanitize_basename(slug))
        m = re.match(r"^(.+)_([a-f0-9]{6})$", slug)
        if m:
            add(m.group(1))

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    try:
        for cid in candidates:
            if client[MONGO_DB].resumes.find_one({"resumeId": cid}, {"_id": 1}):
                return cid
    finally:
        client.close()
    return resume_id


def _line_text(tokens: list[dict]) -> str:
    return " ".join((t.get("token") or "").strip() for t in tokens)


def _group_section_lines(tokens: list[dict]) -> dict[tuple[int, int], list[dict]]:
    lines: dict[tuple[int, int], list[dict]] = {}
    for t in tokens:
        key = (int(t.get("page", 0)), int(t.get("lineIndex", t.get("line_index", 0))))
        lines.setdefault(key, []).append(t)
    for line_tokens in lines.values():
        line_tokens.sort(key=lambda tok: tok.get("tokenIndex", tok.get("token_index", 0)))
    return lines


def _line_similarity(left: str, right: str) -> float:
    left_words = {w for w in re.split(r"\s+", left.strip()) if w}
    right_words = {w for w in re.split(r"\s+", right.strip()) if w}
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def _overlay_by_line_alignment(tokens: list[dict], db_tokens: list[dict]) -> int:
    """Align inference lines to Mongo lines by text when PDF coords drift."""
    inf_lines = _group_section_lines(tokens)
    db_lines = _group_section_lines(db_tokens)
    count = 0

    for (page, _), inf_toks in inf_lines.items():
        inf_text = _line_text(inf_toks)
        best_db: list[dict] | None = None
        best_score = 0.0
        for (db_page, _), db_toks in db_lines.items():
            if db_page != page:
                continue
            score = _line_similarity(inf_text, _line_text(db_toks))
            if score > best_score:
                best_score = score
                best_db = db_toks
        if not best_db or best_score < 0.85:
            continue
        for idx, inf_tok in enumerate(inf_toks):
            if inf_tok.get("_fieldBioLabel"):
                continue
            if idx >= len(best_db):
                break
            lbl = best_db[idx].get("bioLabel")
            if lbl and lbl not in ("O", "B-HEADING", "I-HEADING"):
                inf_tok["_fieldBioLabel"] = lbl
                count += 1
    return count


def _overlay_by_fuzzy_token(tokens: list[dict], db_tokens: list[dict]) -> int:
    """Match tokens on the same page with close x0 and tolerant y0 drift."""
    count = 0
    for t in tokens:
        if t.get("_fieldBioLabel"):
            continue
        inf_tok = (t.get("token") or "").strip()
        if not inf_tok:
            continue
        page = t.get("page")
        x0 = float(t.get("x0", 0))
        y0 = float(t.get("y0", 0))
        best: dict | None = None
        best_dy = 999.0
        for db in db_tokens:
            if db.get("page") != page:
                continue
            db_tok = (db.get("token") or "").strip()
            if db_tok != inf_tok:
                continue
            if abs(float(db.get("x0", 0)) - x0) > 5.0:
                continue
            dy = abs(float(db.get("y0", 0)) - y0)
            if dy < best_dy:
                best_dy = dy
                best = db
        if best is not None and best_dy <= 80.0:
            t["_fieldBioLabel"] = best["bioLabel"]
            count += 1
    return count


def _propagate_merged_labels(tokens: list[dict], db_tokens: list[dict]) -> int:
    """Map split PDF tokens to merged Mongo labels on the same visual row."""
    count = 0
    for t in tokens:
        if t.get("_fieldBioLabel"):
            continue
        inf_tok = (t.get("token") or "").strip()
        if not inf_tok:
            continue
        page = t.get("page")
        y0 = float(t.get("y0", 0))
        x0 = float(t.get("x0", 0))

        for db in db_tokens:
            if db.get("page") != page:
                continue
            if abs(float(db.get("y0", 0)) - y0) > 3.0:
                continue
            db_tok = (db.get("token") or "").strip()
            if not db_tok:
                continue
            if abs(float(db.get("x0", 0)) - x0) <= 1.0 and (
                inf_tok == db_tok
                or db_tok.startswith(inf_tok)
                or inf_tok.startswith(db_tok)
            ):
                t["_fieldBioLabel"] = db["bioLabel"]
                count += 1
                break
            if _is_substring_token_match(inf_tok, db_tok):
                t["_fieldBioLabel"] = db["bioLabel"]
                count += 1
                break
    return count


def overlay_mongo_field_labels(tokens: list[dict], resume_id: str, slug: str | None = None) -> int:
    """Store MongoDB field labels on `_fieldBioLabel` keyed by (page, x0, y0).

    Kept separate from `bioLabel` so boundary inference (B-ENTRY) does not erase them
    before segmentation postprocess reads bio hints.
    """
    try:
        from pymongo import MongoClient
        from core.config import MONGO_DB, MONGO_URI
    except ImportError:
        return 0

    mongo_resume_id = resolve_mongo_resume_id(resume_id, slug)
    client = MongoClient(MONGO_URI)
    doc = client[MONGO_DB].resumes.find_one({"resumeId": mongo_resume_id})
    client.close()
    if not doc:
        return 0

    target_section = tokens[0].get("section") if tokens else None
    db_labeled: list[dict] = []
    db_by_coord: dict[tuple, str] = {}
    db_section_tokens: list[dict] = []
    for t in doc.get("tokens", []):
        if target_section and t.get("section") != target_section:
            continue
        db_section_tokens.append(t)
        lbl = t.get("bioLabel", "O")
        if lbl and lbl not in ("O", "B-HEADING", "I-HEADING"):
            db_by_coord[_coord_key(t)] = lbl
            db_labeled.append({"page": t.get("page"), "x0": t.get("x0"), "y0": t.get("y0"), "token": t.get("token"), "bioLabel": lbl})

    if not db_section_tokens:
        return 0

    count = 0
    for t in tokens:
        lbl = db_by_coord.get(_coord_key(t))
        if lbl:
            t["_fieldBioLabel"] = lbl
            count += 1

    count += _overlay_by_line_alignment(tokens, db_section_tokens)
    count += _overlay_by_fuzzy_token(tokens, db_labeled)
    count += _propagate_merged_labels(tokens, db_labeled)
    return count


def load_mongo_entry_heads(resume_id: str, section: str, slug: str | None = None) -> set[tuple[int, int]]:
    """Return confirmed entry-head lines from MongoDB (training eval uses these for slicing)."""
    try:
        from pymongo import MongoClient
        from core.config import MONGO_DB, MONGO_URI
    except ImportError:
        return set()

    mongo_resume_id = resolve_mongo_resume_id(resume_id, slug)
    client = MongoClient(MONGO_URI)
    doc = client[MONGO_DB].resumes.find_one({"resumeId": mongo_resume_id})
    client.close()
    if not doc:
        return set()

    if section == "EXPERIENCE":
        heads = doc.get("experienceEntryHeads") or []
    elif section == "EDUCATION":
        heads = doc.get("educationEntryHeads") or []
    elif section == "PROJECT":
        heads = doc.get("projectEntryHeads") or []
    else:
        return set()

    return {(int(h["page"]), int(h["lineIndex"])) for h in heads if "page" in h and "lineIndex" in h}
