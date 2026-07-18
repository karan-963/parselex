"""Per-stage timing and memory tracking for inference pipeline runs."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

STAGE_LABELS: dict[str, str] = {
    "extract_tokens": "Token extraction",
    "section_phase1": "Section headings",
    "section_phase2": "Section assignment",
    "education_phase2_divider": "Education boundaries",
    "education_phase1_segment": "Education segmentation",
    "education_phase3_classify": "Education classification",
    "skills_classify": "Skills classification",
    "experience_phase2_divider": "Experience boundaries",
    "experience_phase1_segment": "Experience segmentation",
    "experience_phase3_classify": "Experience classification",
    "project_phase2_divider": "Project boundaries",
    "project_phase1_segment": "Project segmentation",
    "project_phase3_classify": "Project classification",
    "personal_classify": "Personal classification",
    "finalize": "Finalize & structure",
}


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return 0.0


class PipelinePerformanceTracker:
    def __init__(self) -> None:
        self._started_at = time.perf_counter()
        self.stages: list[dict[str, Any]] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        mem_before = _rss_mb()
        t0 = time.perf_counter()
        yield
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        mem_after = _rss_mb()
        self.stages.append(
            {
                "stage": name,
                "label": STAGE_LABELS.get(name, name),
                "durationMs": elapsed_ms,
                "memoryMb": round(mem_after, 1),
                "memoryDeltaMb": round(mem_after - mem_before, 1),
            }
        )

    def summary(self) -> dict[str, Any]:
        total_ms = round((time.perf_counter() - self._started_at) * 1000, 1)
        peak_mem = max((float(s["memoryMb"]) for s in self.stages), default=_rss_mb())
        return {
            "totalDurationMs": total_ms,
            "peakMemoryMb": round(peak_mem, 1),
            "stages": self.stages,
        }
