/**
 * Gate MongoDB ground-truth display to the bundled default reference resume only.
 *
 * Uploaded resumes are pure inference — no MongoDB GT comparison is shown.
 * Ground truth is reserved for the demo reference resume (Karan) whose labels
 * exist in MongoDB. Mirrors training-engine/inference_v2/gt_gate.py.
 */

export const DEFAULT_GT_RESUME_ID = 'Karan';

export const isGtEnabled = (resumeId?: string | null): boolean =>
  !!resumeId && resumeId === DEFAULT_GT_RESUME_ID;
