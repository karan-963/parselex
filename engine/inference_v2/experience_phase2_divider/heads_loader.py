"""Load experience entry heads from MongoDB when available."""

from __future__ import annotations

import os
import sys

_ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from pymongo import MongoClient
from inference_v2.overlay_mongo_labels import resolve_mongo_resume_id
from utils.entry_heads import resolve_entry_head_lines


def load_entry_head_lines(resume_id: str, tokens: list[dict], slug: str | None = None) -> set[tuple[int, int]]:
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGO_DB", "resume-labeling")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        mongo_resume_id = resolve_mongo_resume_id(resume_id, slug)
        doc = client[mongo_db]["resumes"].find_one({"resumeId": mongo_resume_id})
        client.close()
    except Exception:
        return set()
    if not doc:
        return set()
    heads = doc.get("experienceEntryHeads") or []
    if not heads:
        return set()
    reference_tokens = [t for t in doc.get("tokens", []) if t.get("section") == "EXPERIENCE"]
    return resolve_entry_head_lines(tokens, heads, reference_tokens)
