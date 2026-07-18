/** Field classification block report (phase3 segment classification). */

export interface FieldClassificationRow {
  status: string;
  entryKey: string;
  gt: string;
  pred: string;
  confidence?: number;
  text: string;
}

export interface FieldClassificationReport {
  gtSource: string;
  entryHeadSource: string;
  gtEntryHeadLines: { page: number; lineIndex: number }[];
  macroClasses: string[];
  metrics: {
    macroF1ProxyPercent: number;
    blocks: number;
    correct: number;
    errors: number;
  };
  blockRows: FieldClassificationRow[];
}

/** Client fallback when artifact lacks blockClassification (pre-rerun). */
export const buildFieldClassificationReport = (
  _dbResume: unknown,
  artifact: { blockClassification?: FieldClassificationReport },
): FieldClassificationReport | null => artifact.blockClassification ?? null;

export interface PredictedFieldRow {
  segIndex: number;
  label: string;
  confidence: number;
  text: string;
  page: number;
  lineIndex: number;
}

interface PredToken {
  page?: number;
  lineIndex?: number;
  tokenIndex?: number;
  token?: string;
  prediction?: string;
  confidence?: number;
}

/**
 * Group classified tokens into contiguous field segments from model predictions
 * alone (no ground truth). A new segment starts at any `B-*` label or when the
 * field class changes; `O` tokens close the current segment. Used to render the
 * prediction-only view for uploaded resumes where GT comparison is unavailable.
 */
export const buildPredictedFieldReport = (tokens: PredToken[]): PredictedFieldRow[] => {
  const rows: PredictedFieldRow[] = [];
  let current: PredictedFieldRow | null = null;
  let currentClass: string | null = null;

  for (const t of tokens) {
    const pred = t.prediction || 'O';
    if (pred === 'O') {
      current = null;
      currentClass = null;
      continue;
    }
    const cls = pred.replace(/^[BI]-/, '');
    const isBoundary = pred.startsWith('B-');
    const tok = (t.token || '').trim();

    if (!current || isBoundary || cls !== currentClass) {
      current = {
        segIndex: rows.length + 1,
        label: cls,
        confidence: t.confidence ?? 0,
        text: tok,
        page: t.page ?? 0,
        lineIndex: t.lineIndex ?? 0,
      };
      rows.push(current);
      currentClass = cls;
    } else {
      current.text = current.text ? `${current.text} ${tok}` : tok;
    }
  }

  return rows;
};
