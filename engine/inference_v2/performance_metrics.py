"""Aggregate per-section timing, memory, confidence, and accuracy scores."""

from __future__ import annotations

from typing import Any

SECTION_GROUPS: dict[str, dict[str, Any]] = {
    "section_headings": {
        "label": "Section Headings",
        "stages": ["section_phase1", "section_phase2"],
    },
    "profile": {
        "label": "Profile",
        "stages": ["personal_classify"],
        "tokenSection": "PERSONAL",
    },
    "education": {
        "label": "Education",
        "stages": [
            "education_phase2_divider",
            "education_phase1_segment",
            "education_phase3_classify",
        ],
        "tokenSection": "EDUCATION",
    },
    "skills": {
        "label": "Skills",
        "stages": ["skills_classify"],
        "tokenSection": "SKILLS",
    },
    "experience": {
        "label": "Experience",
        "stages": [
            "experience_phase2_divider",
            "experience_phase1_segment",
            "experience_phase3_classify",
        ],
        "tokenSection": "EXPERIENCE",
    },
    "projects": {
        "label": "Projects",
        "stages": [
            "project_phase2_divider",
            "project_phase1_segment",
            "project_phase3_classify",
        ],
        "tokenSection": "PROJECT",
    },
}


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 1)


def _confidence_from_tokens(tokens: list[dict], section: str | None = None) -> float | None:
    confs: list[float] = []
    for tok in tokens:
        if section and tok.get("section") != section:
            continue
        conf = tok.get("confidence")
        if conf is None:
            continue
        try:
            val = float(conf)
        except (TypeError, ValueError):
            continue
        if val > 0:
            confs.append(val)
    if not confs:
        return None
    return round(sum(confs) / len(confs) * 100.0, 1)


def _extract_accuracy_scores(stage_outputs: dict[str, Any]) -> dict[str, float | None]:
    """Map pipeline stage keys to optional GT accuracy percent."""
    scores: dict[str, float | None] = {}

    p1 = stage_outputs.get("section_phase1") or {}
    headings = p1.get("headings") or []
    if headings:
        confs = [float(h["confidence"]) for h in headings if h.get("confidence") is not None]
        scores["section_phase1"] = _mean([c * 100 for c in confs])

    skills = stage_outputs.get("skills_classify") or {}
    metrics = (skills.get("tokenClassification") or {}).get("metrics") or {}
    if metrics.get("evalTokens", 0) > 0:
        scores["skills_classify"] = float(metrics.get("tokenAccuracyPercent", 0))

    for key in (
        "education_phase3_classify",
        "personal_classify",
        "project_phase3_classify",
        "experience_phase3_classify",
    ):
        out = stage_outputs.get(key) or {}
        block_metrics = (out.get("blockClassification") or {}).get("metrics") or {}
        if block_metrics.get("blocks", 0) > 0 or block_metrics.get("segments", 0) > 0:
            score = block_metrics.get("segmentAccuracyPercent")
            if score is None:
                score = block_metrics.get("macroF1ProxyPercent")
            if score is not None:
                scores[key] = float(score)

    for key in ("education_phase2_divider", "experience_phase2_divider", "project_phase2_divider"):
        out = stage_outputs.get(key) or {}
        divider = out.get("entryDividerLines") or {}
        metrics = divider.get("metrics") or {}
        if metrics.get("gtEntryLines", 0) > 0:
            scores[key] = float(metrics.get("fbaPercent", 0))

    for key in (
        "education_phase1_segment",
        "experience_phase1_segment",
        "project_phase1_segment",
    ):
        out = stage_outputs.get(key) or {}
        seg_metrics = (out.get("tokenSegmentation") or {}).get("metrics") or {}
        if seg_metrics.get("evalTokens", 0) > 0:
            scores[key] = float(seg_metrics.get("tokenAccuracyPercent", 0))

    return scores


def _count_section_tokens(tokens: list[dict], section: str) -> int:
    return sum(1 for tok in tokens if tok.get("section") == section)


def _section_is_present(key: str, meta: dict[str, Any], tokens: list[dict], stage_outputs: dict[str, Any]) -> bool:
    if key == "section_headings":
        p1 = stage_outputs.get("section_phase1") or {}
        return bool(p1.get("headings"))

    token_section = meta.get("tokenSection")
    if token_section:
        return _count_section_tokens(tokens, token_section) > 0
    return True


def _section_accuracy_score(stage_names: list[str], stage_scores: dict[str, float | None]) -> float | None:
    vals = [stage_scores[s] for s in stage_names if stage_scores.get(s) is not None]
    return _mean(vals) if vals else None


def enrich_performance_summary(
    summary: dict[str, Any],
    *,
    tokens: list[dict],
    stage_outputs: dict[str, Any],
) -> dict[str, Any]:
    stage_by_name = {s["stage"]: s for s in summary.get("stages", [])}
    stage_accuracy = _extract_accuracy_scores(stage_outputs)

    sections: dict[str, Any] = {}
    section_score_vals: list[float] = []
    section_conf_vals: list[float] = []

    for key, meta in SECTION_GROUPS.items():
        if not _section_is_present(key, meta, tokens, stage_outputs):
            continue

        stage_names: list[str] = meta["stages"]
        stage_rows = [stage_by_name[s] for s in stage_names if s in stage_by_name]
        duration_ms = round(sum(float(r["durationMs"]) for r in stage_rows), 1) if stage_rows else 0.0
        memory_mb = max((float(r["memoryMb"]) for r in stage_rows), default=0.0)

        token_section = meta.get("tokenSection")
        confidence = _confidence_from_tokens(tokens, token_section) if token_section else None
        if confidence is None and key == "section_headings":
            p1 = stage_outputs.get("section_phase1") or {}
            headings = p1.get("headings") or []
            confs = [float(h["confidence"]) * 100 for h in headings if h.get("confidence") is not None]
            confidence = _mean(confs)

        accuracy = _section_accuracy_score(stage_names, stage_accuracy)
        score = accuracy if accuracy is not None else confidence
        score_source = "accuracy" if accuracy is not None else ("confidence" if confidence is not None else None)

        if score is not None:
            section_score_vals.append(score)
        if confidence is not None:
            section_conf_vals.append(confidence)

        sections[key] = {
            "label": meta["label"],
            "present": True,
            "durationMs": duration_ms,
            "memoryMb": round(memory_mb, 1),
            "confidencePercent": confidence,
            "scorePercent": score,
            "scoreSource": score_source,
            "stages": stage_names,
        }

    summary["sections"] = sections
    summary["overallScorePercent"] = _mean(section_score_vals)
    summary["overallConfidencePercent"] = _mean(section_conf_vals)
    return summary
