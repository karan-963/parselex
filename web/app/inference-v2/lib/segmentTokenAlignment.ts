/** Segment artifact GT alignment — handles PDF split vs merged MongoDB tokens. */

const STRUCTURAL_SEG = new Set(['|', '•', '-', '–', '—', '*', '▪', '◦', '■', '·', ',', '✓', '✔', '"']);

export const hasAlphanumeric = (text: string | null): boolean =>
  !!text && /[a-zA-Z0-9]/.test(text);

export const isEvalSegToken = (text: string | null): boolean => {
  const tok = (text ?? '').trim();
  if (!tok || STRUCTURAL_SEG.has(tok)) return false;
  return hasAlphanumeric(tok);
};

export const roundCoord = (n: number) => Math.round(n * 100) / 100;

export const coordKey = (t: { page: number; x0?: number; y0?: number }) =>
  `${t.page}-${roundCoord(t.x0 ?? 0)}-${roundCoord(t.y0 ?? 0)}`;

export const alnumCore = (text: string | null): string => {
  if (!text) return '';
  return text.replace(/^[^a-zA-Z0-9]+/, '').replace(/[^a-zA-Z0-9]+$/, '');
};

/** Map MongoDB bioLabel to skills 5-class BIO (matches training data.py). */
export const mapSkillsLabelTo5Class = (label: string | null | undefined): string => {
  if (!label || label === 'O') return 'O';
  const upper = label.toUpperCase();
  const prefix = upper[0];
  if (!['B', 'I', 'L', 'U'].includes(prefix) || upper.length < 3 || upper[1] !== '-') {
    return 'O';
  }
  const tag = upper.slice(2);
  let targetTag: string | null = null;
  if (tag === 'SKILL') targetTag = 'SKILL';
  else if (tag === 'SKILL_TYPE' || tag === 'SKILL_CAT') targetTag = 'SKILL_TYPE';
  else return 'O';
  if (prefix === 'B' || prefix === 'U') return `B-${targetTag}`;
  if (prefix === 'I' || prefix === 'L') return `I-${targetTag}`;
  return 'O';
};

const sameYBand = (a: { page: number; y0?: number }, b: { y0?: number; page: number }) =>
  a.page === b.page && Math.abs((a.y0 ?? 0) - (b.y0 ?? 0)) <= 3.0;

const isSubstringTokenMatch = (infTok: string, dbTok: string): boolean => {
  if (infTok === dbTok) return true;
  if (!dbTok.includes(infTok)) return false;
  const idx = dbTok.indexOf(infTok);
  const before = idx === 0 ? '' : dbTok[idx - 1];
  const after = idx + infTok.length >= dbTok.length ? '' : dbTok[idx + infTok.length];
  const okBefore = !before || !/[a-zA-Z0-9]/.test(before);
  const okAfter = !after || !/[a-zA-Z0-9]/.test(after);
  return okBefore && okAfter;
};

/** Fuzzy coord match when extract vs Mongo x0/y0 differ by ≤1pt (e.g. Moradabad vs Moradabad)). */
const findFuzzyCoordMatch = (
  inf: any,
  sectionDbTokens: any[],
): any | null => {
  for (const t of sectionDbTokens) {
    if (t.page !== inf.page) continue;
    if (Math.abs((t.x0 ?? 0) - (inf.x0 ?? 0)) > 1.0) continue;
    if (Math.abs((t.y0 ?? 0) - (inf.y0 ?? 0)) > 1.0) continue;
    const infCore = alnumCore(inf.token);
    const dbCore = alnumCore(t.token);
    if (!infCore || !dbCore) continue;
    if (infCore === dbCore || dbCore.startsWith(infCore) || infCore.startsWith(dbCore)) {
      return t;
    }
  }
  return null;
};

/** Resolve GT mongo token for segment rows (coord → fuzzy → merged-line substring). */
export const resolveSegmentGtToken = (
  inf: any,
  dbByCoord: Map<string, any>,
  sectionDbTokens: any[],
): any | null => {
  const primary = dbByCoord.get(coordKey(inf));
  if (primary) return primary;

  const fuzzy = findFuzzyCoordMatch(inf, sectionDbTokens);
  if (fuzzy) return fuzzy;

  const infTok = (inf.token ?? '').trim();
  if (!infTok) return null;

  if (!hasAlphanumeric(infTok)) {
    return (
      sectionDbTokens.find(
        (t) =>
          sameYBand(inf, t) &&
          (t.token ?? '').includes(infTok) &&
          (t.token ?? '').trim() !== infTok,
      ) ?? null
    );
  }

  return (
    sectionDbTokens.find(
      (t) => sameYBand(inf, t) && isSubstringTokenMatch(infTok, (t.token ?? '').trim()),
    ) ?? null
  );
};

export interface SegmentEvalStats {
  total: number;
  matches: number;
  mismatches: number;
  notInDb: number;
  notExtracted: number;
  evalTokens: number;
  accuracy: string;
  headingsMode: boolean;
}

export const computeSegmentEvalStats = (
  rows: Array<{ status: string; token: string | null }>,
  artifactMetrics?: { tokenAccuracyPercent?: number; correct?: number; evalTokens?: number },
): SegmentEvalStats => {
  const evalRows = rows.filter((r) => isEvalSegToken(r.token));
  const matches = evalRows.filter((r) => r.status === '✅ MATCH' || r.status.includes('split')).length;
  const mismatches = evalRows.filter(
    (r) => r.status === '❌ MISMATCH' || r.status.startsWith('❌ MISMATCH'),
  ).length;
  const notInDb = evalRows.filter((r) => r.status === '❌ NOT IN DB' || r.status.startsWith('❌ SECTION MISMATCH')).length;
  const notExtracted = rows.filter((r) => r.status === '❌ NOT EXTRACTED').length;

  const evalTokens = artifactMetrics?.evalTokens ?? evalRows.length;
  const correct = artifactMetrics?.correct ?? matches;
  const accuracy =
    artifactMetrics?.tokenAccuracyPercent != null
      ? artifactMetrics.tokenAccuracyPercent.toFixed(1)
      : evalTokens > 0
        ? ((correct / evalTokens) * 100).toFixed(1)
        : '0';

  return {
    total: evalRows.length,
    matches: correct,
    mismatches,
    notInDb,
    notExtracted,
    evalTokens,
    accuracy,
    headingsMode: true,
  };
};
