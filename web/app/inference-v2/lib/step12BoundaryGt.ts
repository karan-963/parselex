/**
 * Step 12 ground truth — LOCKED.
 *
 * Source: MongoDB `tokens[].bioLabel` (entity labels from /viewer).
 * Maps entity BIO → boundary BIO only.
 */

export const STEP12_GT_SOURCE = 'mongodb.bioLabel→boundary' as const;

/** Entity bioLabel → boundary GT */
export function step12GroundTruthLabel(bioLabel: string | undefined | null): string {
  const bio = bioLabel ?? '';
  if (!bio || bio === 'O' || bio.includes('HEADING')) return 'O';
  if (bio.startsWith('B-') && !bio.includes('PROJ_START')) return 'B-PROJ_START';
  if (bio.startsWith('I-')) return 'I-PROJ_START';
  if (bio.startsWith('B-PROJ_START')) return 'B-PROJ_START';
  return 'O';
}

export function applyStep12GroundTruthLabels(
  tokens: Array<{ bioLabel?: string; [key: string]: unknown }>,
  labelKey = '_temp_gtLabel',
): void {
  for (const t of tokens) {
    (t as Record<string, string>)[labelKey] = step12GroundTruthLabel(t.bioLabel);
  }
}
