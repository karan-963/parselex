"""Gate MongoDB ground-truth usage to the bundled default reference resume only.

Uploaded resumes are pure inference — they must never read MongoDB labels for
slicing, overlays, or comparison reports. Ground truth is reserved for the demo
reference resume (Karan) so its metrics can be shown against known labels.
"""

from __future__ import annotations

DEFAULT_GT_RESUME_ID = "Karan"


def is_gt_enabled(resume_id: str | None, slug: str | None = None) -> bool:
    """True only for the default reference resume (GT available in MongoDB)."""
    from .storage import sanitize_basename

    for candidate in (resume_id, slug):
        if candidate and sanitize_basename(candidate) == DEFAULT_GT_RESUME_ID:
            return True
    return False
