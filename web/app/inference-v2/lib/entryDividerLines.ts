/** Entry section divider lines — GT from MongoDB experienceEntryHeads. */

import { resolveEntrySliceHeadLines } from './entrySliceHeads';

export interface EntryDividerLineRow {
  status: string;
  page: number;
  line: number;
  gt: string;
  pred: string;
  tokenLabels: string;
  text: string;
  confidence: number | null;
}

export interface EntryDividerReport {
  gtSource: string;
  gtEntryLines: { page: number; lineIndex: number }[];
  predEntryLines: { page: number; lineIndex: number }[];
  metrics: {
    fbaPercent: number;
    gtEntryLines: number;
    matched: number;
    missed: number;
    extra: number;
  };
  lineRows: EntryDividerLineRow[];
}

export const resolveEntryHeadLines = (
  heads: { page: number; lineIndex: number }[],
  tokens: { page: number; lineIndex: number }[],
): Set<string> => {
  const lineHasTokens = new Set(tokens.map((t) => `${t.page}-${t.lineIndex}`));
  const resolved = new Set<string>();
  for (const h of heads) {
    const direct = `${h.page}-${h.lineIndex}`;
    if (lineHasTokens.has(direct)) {
      resolved.add(direct);
      continue;
    }
    let globalCounter = -1;
    let lastPage = -1;
    let lastLine = -1;
    const globalToPageLine = new Map<string, string>();
    for (const t of [...tokens].sort(
      (a, b) => a.page - b.page || a.lineIndex - b.lineIndex,
    )) {
      if (t.page !== lastPage || t.lineIndex !== lastLine) {
        globalCounter += 1;
        lastPage = t.page;
        lastLine = t.lineIndex;
        globalToPageLine.set(`${t.page}-${globalCounter}`, `${t.page}-${t.lineIndex}`);
      }
    }
    const mapped = globalToPageLine.get(`${h.page}-${h.lineIndex}`);
    resolved.add(mapped ?? direct);
  }
  return resolved;
};

export const predEntryLinesFromBoundaryArtifact = (
  boundaryTokens: { page: number; lineIndex: number; tokenIndex?: number; token?: string; prediction?: string }[],
): Set<string> => resolveEntrySliceHeadLines(boundaryTokens);

type CoordToken = { page: number; lineIndex: number; tokenIndex?: number; token?: string };

const leadingWords = (
  tokens: CoordToken[],
  page: number,
  lineIndex: number,
): string[] => {
  const text = tokens
    .filter((t) => t.page === page && t.lineIndex === lineIndex)
    .sort((a, b) => (a.tokenIndex ?? 0) - (b.tokenIndex ?? 0))
    .map((t) => (t.token ?? '').trim())
    .join(' ')
    .trim();
  return text.split(/\s+/).filter(Boolean);
};

const wordMatch = (a: string, b: string): boolean => {
  const al = a.toLowerCase();
  const bl = b.toLowerCase();
  return al === bl || al.includes(bl) || bl.includes(al);
};

/**
 * Map a MongoDB head line (labeling line-space) to the matching line in the
 * inference token space by leading-token content. PDF re-extraction shifts line
 * indices, so direct (page,lineIndex) comparison fails; content matching mirrors
 * training-engine utils/entry_heads.py.
 */
const matchHeadByContent = (
  referenceWords: string[],
  targetTokens: CoordToken[],
  page: number,
): string | null => {
  if (!referenceWords.length) return null;
  const lines = new Map<number, string[]>();
  for (const t of targetTokens) {
    if (t.page !== page) continue;
    if (!lines.has(t.lineIndex)) lines.set(t.lineIndex, leadingWords(targetTokens, page, t.lineIndex));
  }
  const minPrefix = Math.min(3, referenceWords.length);
  let bestKey: string | null = null;
  let bestScore = 0;
  for (const [lineIndex, words] of lines) {
    if (!words.length) continue;
    let prefix = 0;
    for (let i = 0; i < Math.min(referenceWords.length, words.length); i += 1) {
      if (wordMatch(referenceWords[i], words[i])) prefix += 1;
      else break;
    }
    const refSet = new Set(referenceWords.map((w) => w.toLowerCase()));
    const gotSet = new Set(words.map((w) => w.toLowerCase()));
    let overlap = 0;
    for (const w of refSet) if (gotSet.has(w)) overlap += 1;
    const ratio = overlap / Math.max(referenceWords.length, 1);
    const score = prefix * 2 + ratio;
    if (prefix >= minPrefix && score > bestScore) {
      bestScore = score;
      bestKey = `${page}-${lineIndex}`;
    }
  }
  return bestKey;
};

/** Resolve MongoDB heads into the inference token line-space via content matching. */
const resolveHeadsToInferenceSpace = (
  heads: { page: number; lineIndex: number }[],
  mongoTokens: CoordToken[],
  inferenceTokens: CoordToken[],
): Set<string> => {
  const resolved = new Set<string>();
  for (const h of heads) {
    const refWords = leadingWords(mongoTokens, h.page, h.lineIndex);
    const matched = matchHeadByContent(refWords, inferenceTokens, h.page);
    resolved.add(matched ?? `${h.page}-${h.lineIndex}`);
  }
  return resolved;
};

const lineText = (
  tokens: { page: number; lineIndex: number; token?: string }[],
  page: number,
  lineIndex: number,
): string =>
  tokens
    .filter((t) => t.page === page && t.lineIndex === lineIndex)
    .map((t) => t.token ?? '')
    .join(' ')
    .trim();

const tokenLabelsOnLine = (
  tokens: { page: number; lineIndex: number; bioLabel?: string; prediction?: string }[],
  page: number,
  lineIndex: number,
  boundaryTokens: { page: number; lineIndex: number; prediction?: string }[],
): string => {
  const fromBoundary = new Set(
    boundaryTokens
      .filter((t) => t.page === page && t.lineIndex === lineIndex)
      .map((t) => t.prediction ?? 'O'),
  );
  if (fromBoundary.size > 0) {
    return [...fromBoundary].sort().join(', ');
  }
  const labels = new Set(
    tokens
      .filter((t) => t.page === page && t.lineIndex === lineIndex)
      .map((t) => t.bioLabel ?? t.prediction ?? 'O'),
  );
  return [...labels].sort().join(', ');
};

/** Confidence of the boundary-label token(s) on a line (e.g. the B-ENTRY token), if any. */
const boundaryConfidenceOnLine = (
  boundaryTokens: { page: number; lineIndex: number; prediction?: string; confidence?: number }[],
  page: number,
  lineIndex: number,
  label: string,
): number | null => {
  const hits = boundaryTokens.filter(
    (t) => t.page === page && t.lineIndex === lineIndex && t.prediction === label && typeof t.confidence === 'number',
  );
  if (!hits.length) return null;
  return Math.min(...hits.map((t) => t.confidence as number));
};

export interface PredictedEntryReport {
  startLabel: string;
  predEntryLines: { page: number; lineIndex: number }[];
  lineRows: EntryDividerLineRow[];
}

/**
 * Build a ground-truth-free view of predicted entry boundaries from the step
 * divider artifact tokens. Used for uploaded resumes (no MongoDB GT) so Step 9
 * still renders the model's predicted job/education/project entry starts.
 */
export const buildPredictedEntryReport = (
  boundaryTokens: { page: number; lineIndex: number; tokenIndex?: number; token?: string; prediction?: string; confidence?: number }[],
  section: 'EXPERIENCE' | 'PROJECT' | 'EDUCATION' = 'EXPERIENCE',
): PredictedEntryReport => {
  const isProject = section === 'PROJECT';
  const isEducation = section === 'EDUCATION';
  const startLabel = isProject ? 'B-PROJ_START' : isEducation ? 'B-EDU_START' : 'B-ENTRY';

  const predLines = isProject
    ? new Set(boundaryTokens.filter((t) => t.prediction === 'B-PROJ_START').map((t) => `${t.page}-${t.lineIndex}`))
    : isEducation
      ? new Set(boundaryTokens.filter((t) => t.prediction === 'B-EDU_START').map((t) => `${t.page}-${t.lineIndex}`))
      : predEntryLinesFromBoundaryArtifact(boundaryTokens);

  const keys = [...predLines].sort((a, b) => {
    const [ap, al] = a.split('-').map(Number);
    const [bp, bl] = b.split('-').map(Number);
    return ap - bp || al - bl;
  });

  const lineRows: EntryDividerLineRow[] = keys.map((key) => {
    const [page, line] = key.split('-').map(Number);
    return {
      status: '◆ ENTRY START',
      page,
      line,
      gt: '',
      pred: startLabel,
      tokenLabels: tokenLabelsOnLine(boundaryTokens, page, line, boundaryTokens),
      text: lineText(boundaryTokens, page, line).slice(0, 140),
      confidence: boundaryConfidenceOnLine(boundaryTokens, page, line, startLabel),
    };
  });

  return {
    startLabel,
    predEntryLines: keys.map((k) => {
      const [page, lineIndex] = k.split('-').map(Number);
      return { page, lineIndex };
    }),
    lineRows,
  };
};

export const buildEntryDividerReport = (
  dbResume: {
    experienceEntryHeads?: { page: number; lineIndex: number }[];
    projectEntryHeads?: { page: number; lineIndex: number }[];
    educationEntryHeads?: { page: number; lineIndex: number }[];
    tokens?: any[];
  },
  boundaryTokens: { page: number; lineIndex: number; prediction?: string; confidence?: number }[],
  artifactEntryDivider?: EntryDividerReport | null,
  section: 'EXPERIENCE' | 'PROJECT' | 'EDUCATION' = 'EXPERIENCE',
): EntryDividerReport | null => {
  if (artifactEntryDivider?.lineRows?.length) {
    return artifactEntryDivider;
  }

  const isProject = section === 'PROJECT';
  const isEducation = section === 'EDUCATION';
  const heads = isProject
    ? (dbResume.projectEntryHeads ?? [])
    : isEducation
      ? (dbResume.educationEntryHeads ?? [])
      : (dbResume.experienceEntryHeads ?? []);
  const sectionTokens = (dbResume.tokens ?? []).filter((t: any) => {
    if (isProject) {
      return (t.section === 'PROJECT' || t.section === 'PROJECTS')
        && t.bioLabel !== 'B-HEADING' && t.bioLabel !== 'I-HEADING';
    }
    if (isEducation) {
      return t.section === 'EDUCATION'
        && t.bioLabel !== 'B-HEADING' && t.bioLabel !== 'I-HEADING';
    }
    return t.section === 'EXPERIENCE' && t.bioLabel !== 'B-HEADING' && t.bioLabel !== 'I-HEADING';
  });
  if (!heads.length || !sectionTokens.length) return null;

  // GT heads are stored in the MongoDB labeling line-space. Inference re-extracts the
  // PDF with different line indices, so map heads into the inference token space by
  // content before comparing with predicted boundary lines. Fall back to MongoDB-space
  // resolution when inference tokens are unavailable.
  const gtLines = boundaryTokens.length
    ? resolveHeadsToInferenceSpace(heads, sectionTokens, boundaryTokens)
    : resolveEntryHeadLines(heads, sectionTokens);
  // Use inference tokens for line text/labels so GT and pred rows align in one space.
  const displayTokens = boundaryTokens.length ? boundaryTokens : sectionTokens;
  const predLines = isProject
    ? new Set(
        boundaryTokens
          .filter((t) => t.prediction === 'B-PROJ_START')
          .map((t) => `${t.page}-${t.lineIndex}`),
      )
    : isEducation
      ? new Set(
          boundaryTokens
            .filter((t) => t.prediction === 'B-EDU_START')
            .map((t) => `${t.page}-${t.lineIndex}`),
        )
      : predEntryLinesFromBoundaryArtifact(boundaryTokens);

  const gtLabel = isProject ? 'B-PROJ_START' : isEducation ? 'B-EDU_START' : 'B-ENTRY';
  const gtSource = isProject
    ? 'mongodb.projectEntryHeads'
    : isEducation
      ? 'mongodb.educationEntryHeads'
      : 'mongodb.experienceEntryHeads';

  const allKeys = [...new Set([...gtLines, ...predLines])].sort((a, b) => {
    const [ap, al] = a.split('-').map(Number);
    const [bp, bl] = b.split('-').map(Number);
    return ap - bp || al - bl;
  });

  let matched = 0;
  let missed = 0;
  let extra = 0;
  const lineRows: EntryDividerLineRow[] = [];

  for (const key of allKeys) {
    const [page, line] = key.split('-').map(Number);
    const gt = gtLines.has(key);
    const pred = predLines.has(key);
    let status: string;
    if (gt && pred) {
      status = '✅ MATCH';
      matched += 1;
    } else if (gt) {
      status = '❌ MISSED';
      missed += 1;
    } else {
      status = '⚠️ EXTRA';
      extra += 1;
    }
    lineRows.push({
      status,
      page,
      line,
      gt: gt ? gtLabel : '',
      pred: pred ? gtLabel : '',
      tokenLabels: tokenLabelsOnLine(displayTokens, page, line, boundaryTokens),
      text: lineText(displayTokens, page, line).slice(0, 140),
      confidence: pred ? boundaryConfidenceOnLine(boundaryTokens, page, line, gtLabel) : null,
    });
  }

  const gtTotal = gtLines.size;
  const predTotal = predLines.size;
  const recall = gtTotal > 0 ? matched / gtTotal : predTotal === 0 ? 1 : 0;
  const precision = predTotal > 0 ? matched / predTotal : gtTotal === 0 ? 1 : 0;
  const fba = recall + precision > 0 ? (2 * recall * precision) / (recall + precision) * 100 : 0;

  return {
    gtSource,
    gtEntryLines: [...gtLines].map((k) => {
      const [page, lineIndex] = k.split('-').map(Number);
      return { page, lineIndex };
    }),
    predEntryLines: [...predLines].map((k) => {
      const [page, lineIndex] = k.split('-').map(Number);
      return { page, lineIndex };
    }),
    metrics: {
      fbaPercent: Math.round(fba * 100) / 100,
      gtEntryLines: gtTotal,
      matched,
      missed,
      extra,
    },
    lineRows,
  };
};
