import type { TrainingEvalToken } from './parseTrainingReport';

export interface ArtifactToken {
  page: number;
  lineIndex: number;
  tokenIndex: number;
  token: string;
  prediction: string;
  pageLineToken?: string;
}

export interface ArtifactPayload {
  stage?: string;
  title?: string;
  trainingPipeline?: string;
  labelField?: string;
  section?: string;
  tokens?: ArtifactToken[];
}

export interface CompareRow {
  key: string;
  pageLineToken: string;
  token: string;
  inference: string;
  groundTruth: string;
  match: boolean;
  bioLabel?: string;
}

export interface CompareSummary {
  matched: number;
  total: number;
  pct: number;
  trainingMatched?: number;
  trainingTotal?: number;
  trainingPct?: number;
  gtSource: string;
}

export function tokenKey(page: number, line: number, tok: number): string {
  return `${page}-${line}-${tok}`;
}

export function normalizeArtifactTokens(data: unknown): ArtifactPayload | null {
  if (!data || typeof data !== 'object') return null;
  const payload = data as ArtifactPayload;
  if (!Array.isArray(payload.tokens)) return null;
  return payload;
}

export function buildMongoGroundTruth(
  tokens: Array<{
    page: number;
    lineIndex: number;
    tokenIndex: number;
    token: string;
    bioLabel?: string;
    section?: string;
  }>,
  section: string,
): TrainingEvalToken[] {
  return tokens
    .filter((t) => t.section === section)
    .filter((t) => t.bioLabel !== 'B-HEADING' && t.bioLabel !== 'I-HEADING')
    .map((t) => ({
      page: t.page,
      lineIndex: t.lineIndex,
      tokenIndex: t.tokenIndex,
      token: t.token,
      bioLabel: t.bioLabel,
      groundTruth: t.bioLabel ?? 'O',
    }));
}

function lineKey(page: number, line: number): string {
  return `${page}-${line}`;
}

/** Match tokens within a line by sequential order (indices differ between inference vs training). */
function buildLineOrderGtMap(
  groundTruth: TrainingEvalToken[],
): Map<string, TrainingEvalToken[]> {
  const byLine = new Map<string, TrainingEvalToken[]>();
  for (const gt of groundTruth) {
    const lk = lineKey(gt.page, gt.lineIndex);
    const list = byLine.get(lk) ?? [];
    list.push(gt);
    byLine.set(lk, list);
  }
  for (const list of byLine.values()) {
    list.sort((a, b) => a.tokenIndex - b.tokenIndex);
  }
  return byLine;
}

function dominantBoundaryLabel(predictions: string[]): string {
  if (predictions.some((p) => p === 'B-ENTRY')) return 'B-ENTRY';
  if (predictions.some((p) => p === 'I-ENTRY')) return 'I-ENTRY';
  return 'O';
}

export function buildCompareRows(
  inference: ArtifactToken[],
  groundTruth: TrainingEvalToken[],
  options?: { lineLevel?: boolean },
): CompareRow[] {
  if (options?.lineLevel) {
    const gtByLine = new Map<string, TrainingEvalToken>();
    for (const gt of groundTruth) {
      gtByLine.set(lineKey(gt.page, gt.lineIndex), gt);
    }

    const infByLine = new Map<string, ArtifactToken[]>();
    for (const inf of inference) {
      const lk = lineKey(inf.page, inf.lineIndex);
      const list = infByLine.get(lk) ?? [];
      list.push(inf);
      infByLine.set(lk, list);
    }

    const rows: CompareRow[] = [];
    for (const [lk, gt] of gtByLine) {
      const lineTokens = infByLine.get(lk) ?? [];
      const infLabel = dominantBoundaryLabel(lineTokens.map((t) => t.prediction));
      const [page, line] = lk.split('-').map(Number);
      rows.push({
        key: `${lk}-boundary`,
        pageLineToken: `P${page} L${line}`,
        token: lineTokens[0]?.token ?? gt.token,
        inference: infLabel,
        groundTruth: gt.groundTruth,
        match: infLabel === gt.groundTruth,
      });
    }
    return rows.sort((a, b) => a.pageLineToken.localeCompare(b.pageLineToken));
  }

  const gtByLine = buildLineOrderGtMap(groundTruth);
  const infSorted = [...inference].sort(
    (a, b) =>
      a.page - b.page || a.lineIndex - b.lineIndex || a.tokenIndex - b.tokenIndex,
  );

  const lineCursor = new Map<string, number>();
  const rows: CompareRow[] = [];

  for (const inf of infSorted) {
    const lk = lineKey(inf.page, inf.lineIndex);
    const lineGts = gtByLine.get(lk) ?? [];
    const pos = lineCursor.get(lk) ?? 0;
    const gt = lineGts[pos];
    lineCursor.set(lk, pos + 1);

    const gtLabel = gt?.groundTruth ?? '—';
    const match = gt ? inf.prediction === gt.groundTruth : false;

    rows.push({
      key: `${lk}-${pos}`,
      pageLineToken:
        inf.pageLineToken ??
        `P${inf.page} L${inf.lineIndex} T${gt?.tokenIndex ?? inf.tokenIndex}`,
      token: inf.token,
      inference: inf.prediction,
      groundTruth: gtLabel,
      match: gt ? match : false,
      bioLabel: gt?.bioLabel,
    });
  }

  return rows;
}

export function summarizeCompare(rows: CompareRow[], gtSource: string): CompareSummary {
  const comparable = rows.filter((r) => r.groundTruth !== '—');
  const matched = comparable.filter((r) => r.match).length;
  const total = comparable.length;
  return {
    matched,
    total,
    pct: total > 0 ? (matched / total) * 100 : 0,
    gtSource,
  };
}

export function summarizeTrainingEval(
  tokens: TrainingEvalToken[],
): Pick<CompareSummary, 'trainingMatched' | 'trainingTotal' | 'trainingPct'> {
  const withPred = tokens.filter((t) => t.trainingPred != null);
  const matched = withPred.filter((t) => t.trainingMatch).length;
  const total = withPred.length;
  return {
    trainingMatched: matched,
    trainingTotal: total,
    trainingPct: total > 0 ? (matched / total) * 100 : undefined,
  };
}
