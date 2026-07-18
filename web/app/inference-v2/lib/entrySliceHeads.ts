/** Primary job-entry slice heads (mirrors training-engine entry_slice_heads.py). */

const MAIN_ENTRY_BULLETS = new Set(['•', '●', '▪']);
const SUB_ENTRY_BULLETS = new Set(['◦', '∗', '*', '·']);
const STRUCTURAL = new Set(['"', "'", ',', '|', '(', ')']);
const MONTH_RE =
  /\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)/i;
const YEAR_RE = /\b(19|20)\d{2}\b/;
const APOSTROPHE_YEAR_RE = /[''\u2018\u2019\u2032]\s*(?:\d{4}|\d{2})\b/;
const DATE_END_RE = /\b(present|current|ongoing|now)\b/i;

type LineToken = {
  page: number;
  lineIndex: number;
  tokenIndex?: number;
  token?: string;
  prediction?: string;
  bioLabel?: string;
};

const predOf = (t: LineToken) => t.prediction ?? t.bioLabel ?? 'O';

const firstMeaningful = (line: LineToken[]) => {
  for (const t of line) {
    const tok = (t.token ?? '').trim();
    if (tok && !STRUCTURAL.has(tok)) return tok;
  }
  return '';
};

/** Prediction of the first non-structural token (mirrors _first_meaningful_pred). */
const firstMeaningfulPred = (line: LineToken[]): string | null => {
  for (const t of line) {
    const tok = (t.token ?? '').trim();
    if (tok && !STRUCTURAL.has(tok)) return predOf(t);
  }
  return null;
};

const SECTION_HEADING_TEXT = new Set([
  'experience',
  'work experience',
  'professional experience',
  'employment',
]);

const isSectionHeadingLine = (line: LineToken[]) =>
  SECTION_HEADING_TEXT.has(
    line.map((t) => t.token ?? '').join(' ').trim().toLowerCase(),
  );

const lineHasBEntry = (line: LineToken[]) => line.some((t) => predOf(t) === 'B-ENTRY');

const hasDateAnchor = (text: string) =>
  YEAR_RE.test(text) ||
  APOSTROPHE_YEAR_RE.test(text) ||
  DATE_END_RE.test(text) ||
  MONTH_RE.test(text);

const isDateOnlyLine = (line: LineToken[]) => {
  const text = line.map((t) => t.token ?? '').join(' ').trim();
  const first = firstMeaningful(line);
  if (!first) return false;
  if ((first === '(' || first === '|') && hasDateAnchor(text)) return true;
  return hasDateAnchor(text) && !line.some((t) => {
    const tok = (t.token ?? '').trim();
    return tok.length > 3 && !MONTH_RE.test(tok) && !YEAR_RE.test(tok) && !APOSTROPHE_YEAR_RE.test(tok);
  });
};

const isCompanyOrDateContinuation = (line: LineToken[]) => {
  const first = firstMeaningful(line);
  if (!first || MAIN_ENTRY_BULLETS.has(first)) return false;
  if (first === '(' || first === '|') return true;
  const lower = line.map((t) => t.token ?? '').join(' ').toLowerCase();
  if (hasDateAnchor(lower)) return true;
  return ['pvt', 'ltd', 'inc', 'llc', 'limited'].some((w) => lower.includes(w));
};

/** Lines that start a new experience job block for phrase segmentation. */
export const resolveEntrySliceHeadLines = (
  boundaryTokens: LineToken[],
): Set<string> => {
  const byLine = new Map<string, LineToken[]>();
  for (const t of boundaryTokens) {
    const key = `${t.page}-${t.lineIndex}`;
    if (!byLine.has(key)) byLine.set(key, []);
    byLine.get(key)!.push(t);
  }

  const heads = new Set<string>();
  for (const [key, raw] of byLine) {
    const line = [...raw].sort((a, b) => (a.tokenIndex ?? 0) - (b.tokenIndex ?? 0));
    if (!lineHasBEntry(line)) continue;
    if (isDateOnlyLine(line)) continue;
    if (SUB_ENTRY_BULLETS.has(firstMeaningful(line))) continue;
    if (isCompanyOrDateContinuation(line)) continue;
    if (isSectionHeadingLine(line)) continue;
    // A line starts an entry when its first meaningful token is predicted B-ENTRY —
    // whether that token is a bullet or a plain job title (mirrors Python
    // resolve_entry_slice_heads). Bullet-first lines were the only case handled before,
    // which dropped title-first entries like "ML Engineer …".
    if (firstMeaningfulPred(line) === 'B-ENTRY') {
      heads.add(key);
    }
  }
  return heads;
};
