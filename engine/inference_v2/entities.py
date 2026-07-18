"""Structured entity extraction from classified tokens."""

from __future__ import annotations

import re
from collections import defaultdict


def extract_entities(tokens: list[dict]) -> list[tuple[str, str]]:
    entities: list[tuple[str, str]] = []
    current_entity: list[str] = []
    current_label: str | None = None

    for t in tokens:
        bio = t.get("bioLabel", "O")
        if bio == "O":
            if current_entity:
                entities.append((current_label or "O", " ".join(current_entity)))
                current_entity = []
                current_label = None
            continue

        if bio.startswith("B-"):
            if current_entity:
                entities.append((current_label or "O", " ".join(current_entity)))
            current_label = bio[2:]
            current_entity = [t.get("token", "")]
        elif bio.startswith("I-"):
            label = bio[2:]
            if label == current_label:
                current_entity.append(t.get("token", ""))
            else:
                if current_entity:
                    entities.append((current_label or "O", " ".join(current_entity)))
                current_label = label
                current_entity = [t.get("token", "")]

    if current_entity:
        entities.append((current_label or "O", " ".join(current_entity)))

    cleaned: list[tuple[str, str]] = []
    for label, val in entities:
        if label in ("GITHUB", "LINKEDIN", "OTHER_LINK"):
            # URLs never contain whitespace — token splitting (e.g. "https" / ":"
            # / "//github.com/x") shouldn't leave gaps in the reassembled link.
            val = re.sub(r"\s+", "", val)
        else:
            val = re.sub(r"\s+([,.:;!?])", r"\1", val)
        cleaned.append((label, val))
    return cleaned


def _education_row_marks(entry_tokens: list[dict]) -> list[str]:
    """Collect marks-column values from unlabeled O tokens (e.g. 62.8%, split 69 + %)."""
    marks: list[str] = []
    pending_num = ""
    for token in sorted(entry_tokens, key=lambda item: (float(item.get("x0", 0)), float(item.get("y0", 0)))):
        if (token.get("bioLabel") or "O") != "O":
            pending_num = ""
            continue
        text = (token.get("token") or "").strip()
        if not text:
            continue
        if "%" in text:
            marks.append(f"{pending_num}{text}".strip())
            pending_num = ""
            continue
        if text.isdigit() or re.fullmatch(r"\d+\.?\d*", text):
            pending_num = text
            continue
        pending_num = ""
    return marks


def _education_head_lines(edu_tokens: list[dict], resume_id: str = "") -> set[tuple[int, int]]:
    heads: set[tuple[int, int]] = set()
    for token in edu_tokens:
        if token.get("_eduEntryHead") or token.get("bioLabel") == "B-EDU_START":
            heads.add((int(token["page"]), int(token.get("lineIndex", 0))))
    if not heads and resume_id:
        from inference_v2.education_phase1_segment.entry_slice_heads import resolve_education_entry_heads

        heads = resolve_education_entry_heads(resume_id, edu_tokens)
    return heads


def _group_education_tokens(edu_tokens: list[dict], resume_id: str = "") -> list[list[dict]]:
    head_lines = _education_head_lines(edu_tokens, resume_id)
    if not head_lines:
        return [edu_tokens]

    from inference_v2.education_phase2_divider.entry_groups import group_education_tokens_by_heads

    return group_education_tokens_by_heads(edu_tokens, head_lines)


def build_entities_dict(tokens: list[dict], resume_id: str = "") -> dict:
    entities: dict = defaultdict(list)

    heading_buf: list[str] = []
    for t in tokens:
        lbl = t.get("bioLabel", "O")
        if lbl == "B-HEADING":
            if heading_buf:
                entities["SECTION_HEADINGS"].append(" ".join(heading_buf))
            heading_buf = [t["token"]]
        elif lbl == "I-HEADING":
            heading_buf.append(t["token"])
        else:
            if heading_buf:
                entities["SECTION_HEADINGS"].append(" ".join(heading_buf))
                heading_buf = []
    if heading_buf:
        entities["SECTION_HEADINGS"].append(" ".join(heading_buf))

    seen_h: set[str] = set()
    deduped: list[str] = []
    for h in entities["SECTION_HEADINGS"]:
        key = h.upper()
        if key not in seen_h:
            seen_h.add(key)
            deduped.append(h)
    entities["SECTION_HEADINGS"] = deduped

    section_tokens: dict[str, list] = defaultdict(list)
    for t in tokens:
        section_tokens[t.get("section", "NONE")].append(t)

    if "PERSONAL" in section_tokens:
        for label, val in extract_entities(section_tokens["PERSONAL"]):
            entities["PERSONAL"].append({"label": label, "value": val})

    if "SUMMARY" in section_tokens:
        # Pure heuristic — no classifier for this section. Join the section's
        # non-heading tokens in reading order into one paragraph.
        summary_toks = [
            t for t in section_tokens["SUMMARY"]
            if t.get("bioLabel") not in ("B-HEADING", "I-HEADING")
        ]
        summary_text = " ".join((t.get("token") or "").strip() for t in summary_toks if (t.get("token") or "").strip())
        summary_text = re.sub(r"\s+([,.:;!?])", r"\1", summary_text).strip()
        if summary_text:
            entities["SUMMARY"] = summary_text

    if "EDUCATION" in section_tokens:
        edu = [t for t in section_tokens["EDUCATION"] if t.get("bioLabel") not in ("B-HEADING", "I-HEADING")]
        education_entries: list[list[dict[str, str]]] = []
        for entry_tokens in _group_education_tokens(edu, resume_id):
            entry_entities: list[dict[str, str]] = []
            for label, val in extract_entities(entry_tokens):
                item = {"label": label, "value": val}
                entities["EDUCATION"].append(item)
                entry_entities.append(item)
            for mark in _education_row_marks(entry_tokens):
                item = {"label": "GPA", "value": mark}
                entities["EDUCATION"].append(item)
                entry_entities.append(item)
            if entry_entities:
                education_entries.append(entry_entities)
        if education_entries:
            entities["EDUCATION_ENTRIES"] = education_entries

    if "EXPERIENCE" in section_tokens:
        exp = [t for t in section_tokens["EXPERIENCE"] if t.get("bioLabel") not in ("B-HEADING", "I-HEADING")]
        from inference_v2.experience_phase1_segment.entry_groups import group_experience_tokens_by_heads

        experience_entries: list[list[dict[str, str]]] = []
        for entry_tokens in group_experience_tokens_by_heads(exp, resume_id):
            entry_entities: list[dict[str, str]] = []
            for label, val in extract_entities(entry_tokens):
                item = {"label": label, "value": val}
                entities["EXPERIENCE"].append(item)
                entry_entities.append(item)
            if entry_entities:
                experience_entries.append(entry_entities)
        if experience_entries:
            entities["EXPERIENCE_ENTRIES"] = experience_entries

    proj_sec = "PROJECT" if "PROJECT" in section_tokens else "PROJECTS"
    if proj_sec in section_tokens:
        proj = [t for t in section_tokens[proj_sec] if t.get("bioLabel") not in ("B-HEADING", "I-HEADING")]
        from inference_v2.project_phase2_divider.entry_groups import group_project_tokens_by_heads

        project_entries: list[list[dict[str, str]]] = []
        for entry_tokens in group_project_tokens_by_heads(proj, resume_id):
            entry_entities: list[dict[str, str]] = []
            for label, val in extract_entities(entry_tokens):
                item = {"label": label, "value": val}
                entities["PROJECTS"].append(item)
                entry_entities.append(item)
            if entry_entities:
                project_entries.append(entry_entities)
        if project_entries:
            entities["PROJECT_ENTRIES"] = project_entries

    if "SKILLS" in section_tokens:
        skill_toks = [
            t for t in section_tokens["SKILLS"]
            if t.get("bioLabel") not in ("O", "B-HEADING", "I-HEADING")
            and "SKILL_TYPE" not in t.get("bioLabel", "")
        ]
        ents = extract_entities(skill_toks)
        entities["SKILLS"] = list(dict.fromkeys(val for _, val in ents))

    return dict(entities)
