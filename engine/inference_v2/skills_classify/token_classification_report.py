"""Token-level classification report (mirrors skills/reports/minilm/per_resume/*.md)."""

from __future__ import annotations

import os
from collections import defaultdict

from .config import LABEL_LIST
from .data_utils import alnum_core, coord_key, is_evaluable_token, is_heading_line
from .training_bridge import load_training_helpers
from ..gt_gate import is_gt_enabled


def _mongo_skills_lookups(mongo_tokens: list[dict]) -> tuple[dict, dict[tuple[int, int], list[dict]]]:
    by_coord = {coord_key(token): token for token in mongo_tokens}
    by_line: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for token in mongo_tokens:
        by_line[(int(token.get("page", 0)), int(token.get("lineIndex", 0)))].append(token)
    return by_coord, dict(by_line)


def _resolve_gt_token(
    token: dict,
    mongo_by_coord: dict,
    mongo_by_line: dict[tuple[int, int], list[dict]],
) -> dict | None:
    match = mongo_by_coord.get(coord_key(token))
    if match:
        return match

    line_key = (int(token.get("page", 0)), int(token.get("lineIndex", 0)))
    line_tokens = mongo_by_line.get(line_key, [])
    token_core = alnum_core(token.get("token", ""))
    if token_core:
        for candidate in line_tokens:
            if alnum_core(candidate.get("token", "")) == token_core:
                return candidate

    token_text = (token.get("token") or "").strip()
    if token_text and not alnum_core(token_text):
        for candidate in line_tokens:
            candidate_text = (candidate.get("token") or "").strip()
            if candidate_text.startswith(token_text) and candidate_text != token_text:
                return candidate

    return None


def build_token_classification_report(
    resume_id: str,
    skills_tokens: list[dict],
    pred_labels: list[str],
) -> dict:
    helpers = load_training_helpers()
    map_label = helpers["map_label_to_5class"]

    mongo_by_coord: dict = {}
    mongo_by_line: dict[tuple[int, int], list[dict]] = {}
    if is_gt_enabled(resume_id):
        try:
            from pymongo import MongoClient

            mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
            mongo_db = os.getenv("MONGO_DB", "resume-labeling")
            from ..overlay_mongo_labels import resolve_mongo_resume_id

            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            doc = client[mongo_db]["resumes"].find_one({"resumeId": resolve_mongo_resume_id(resume_id)})
            client.close()
            if doc:
                mongo_skills = [
                    token for token in doc.get("tokens", [])
                    if token.get("section") == "SKILLS"
                    and token.get("bioLabel") not in ("B-HEADING", "I-HEADING")
                ]
                mongo_by_coord, mongo_by_line = _mongo_skills_lookups(mongo_skills)
        except Exception:
            pass

    line_map: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for token in skills_tokens:
        line_map[(int(token.get("page", 0)), int(token.get("lineIndex", 0)))].append(token)

    scored: list[dict] = []
    correct = 0
    total = 0

    for token, pred in zip(skills_tokens, pred_labels):
        token_text = token.get("token", "")
        page = int(token.get("page", 0))
        line_index = int(token.get("lineIndex", 0))
        if is_heading_line(line_map[(page, line_index)], token_text):
            continue
        if not is_evaluable_token(token_text):
            continue

        gt_token = _resolve_gt_token(token, mongo_by_coord, mongo_by_line)
        gt = map_label((gt_token or {}).get("bioLabel", "O") or "O") if gt_token else None
        if gt is None:
            continue

        is_match = gt == pred
        if is_match:
            correct += 1
        total += 1
        scored.append({
            "status": "✅" if is_match else "❌",
            "page": page,
            "lineIndex": line_index,
            "tokenIndex": token.get("tokenIndex", 0),
            "gt": gt,
            "pred": pred,
            "text": (token_text or "")[:80],
        })

    accuracy = (correct / total * 100.0) if total else 0.0
    return {
        "gtSource": "mongodb.tokens.bioLabel (coord + line-text match, 5-class map)",
        "trainingReport": "skills/reports/minilm/per_resume/*.md",
        "labels": LABEL_LIST,
        "metrics": {
            "tokenAccuracyPercent": round(accuracy, 2),
            "evalTokens": total,
            "correct": correct,
            "errors": total - correct,
        },
        "tokenRows": scored,
    }
