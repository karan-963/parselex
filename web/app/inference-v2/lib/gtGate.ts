/**
 * Gate MongoDB ground-truth display to the bundled default reference resume only.
 *
 * Uploaded resumes are pure inference — no MongoDB GT comparison is shown.
 * Ground truth is reserved for the demo reference resume (Karan) whose labels
 * exist in MongoDB. Mirrors training-engine/inference_v2/gt_gate.py.
 */

export const DEFAULT_GT_RESUME_ID = 'Karan';

// Off by default — client deployments have no MongoDB, so ground-truth
// comparison (columns, "Refresh GT" buttons, MongoDB-labeled messaging) is
// dev/internal-only. Set NEXT_PUBLIC_ENABLE_GT=true locally to see it.
export const GT_ENABLED = process.env.NEXT_PUBLIC_ENABLE_GT === 'true';

export const isGtEnabled = (resumeId?: string | null): boolean =>
  GT_ENABLED && !!resumeId && resumeId === DEFAULT_GT_RESUME_ID;
