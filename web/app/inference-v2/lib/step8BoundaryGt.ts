/**
 * Experience boundary ground truth — LOCKED.
 *
 * Source: MongoDB `tokens[].bioLabel` (entity labels from /viewer).
 * Maps entity BIO → boundary BIO only. No label shifting or punctuation normalization.
 *
 * Used for step 9 (`9_experience_boundaries.json`) token-level GT.
 * Do NOT use `experienceEntryHeads` here — that is entry-line FBA GT only.
 * Do not duplicate this logic elsewhere; import from this file.
 */

export const STEP8_GT_SOURCE = 'mongodb.bioLabel→boundary' as const;

/** Entity bioLabel → boundary GT (matches scratch/export_mongo_boundary_gt.py). */
export function step8GroundTruthLabel(bioLabel: string | undefined | null): string {
  const bio = bioLabel ?? '';
  if (!bio || bio === 'O' || bio.includes('HEADING')) return 'O';
  if (bio.startsWith('B-') && !bio.includes('ENTRY')) return 'B-ENTRY';
  if (bio.startsWith('I-')) return 'I-ENTRY';
  if (bio.startsWith('B-ENTRY')) return 'B-ENTRY';
  return 'O';
}

export function applyStep8GroundTruthLabels(
  tokens: Array<{ bioLabel?: string; [key: string]: unknown }>,
  labelKey = '_temp_gtLabel',
): void {
  for (const t of tokens) {
    (t as Record<string, string>)[labelKey] = step8GroundTruthLabel(t.bioLabel);
  }
}
