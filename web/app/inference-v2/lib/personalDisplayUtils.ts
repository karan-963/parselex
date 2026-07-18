/** Display-time cleanup for personal/profile fields in inference-v2 UI. */

const STRUCTURAL_PREFIX_RE =
  /^(?:(?:e-?mail|mail|mobile(?:\s*no\.?)?|phone(?:\s*number)?|tel(?:ephone)?|contact|github(?:\s*link)?|linkedin|website|blog|address)\s*[:.\-–—|]*\s*)+/gi;

const TRAILING_LABEL_RE = /\s*(?:e-?mail|mobile(?:\s*no\.?)?|phone(?:\s*number)?)\s*[-:]*\s*$/gi;

function normalizeSpacedUrl(value: string): string {
  return value
    .replace(/https?\s*:\s*\/\s*\/\s*/gi, (match) =>
      match.toLowerCase().includes('https') ? 'https://' : 'http://',
    )
    .replace(/\s+/g, ' ')
    .trim();
}

function formatGithubUrl(raw: string): string {
  const compact = raw.replace(/\s+/g, '');
  const urlMatch = compact.match(/(?:https?:\/\/)?(?:www\.)?github\.com\/[\w.-]+/i);
  if (urlMatch) {
    const path = urlMatch[0].replace(/^https?:\/\//i, '').replace(/^www\./i, '');
    return `https://${path}`;
  }

  let stripped = raw.replace(STRUCTURAL_PREFIX_RE, '').trim();
  stripped = stripped.replace(/^github\s*link\s*[-:–—|]+\s*/i, '').trim();

  const bareHandle = stripped.match(/^[@/]?([\w.-]+)$/);
  if (bareHandle) {
    return `https://github.com/${bareHandle[1]}`;
  }

  const parts = stripped.split(/[-–—:|/]+/).map((part) => part.trim()).filter(Boolean);
  const handle = parts[parts.length - 1]?.replace(/^@/, '');
  if (handle && /^[\w.-]+$/.test(handle)) {
    return `https://github.com/${handle}`;
  }

  return stripped;
}

export function cleanPersonalDisplayValue(fieldKey: string, raw: string): string {
  if (!raw) return '';

  const key = fieldKey.toLowerCase();
  let value = normalizeSpacedUrl(raw);

  if (key === 'github') {
    return formatGithubUrl(value);
  }

  if (key === 'phone' || key === 'email' || key === 'linkedin' || key === 'link') {
    value = value.replace(STRUCTURAL_PREFIX_RE, '').trim();
    value = value.replace(TRAILING_LABEL_RE, '').trim();
    return value;
  }

  if (key === 'location') {
    value = value.replace(TRAILING_LABEL_RE, '').trim();
    value = value.replace(/\s*e-?mail\s*[-:]*\s*$/i, '').trim();
    return value;
  }

  return value;
}

/** Map B-EMAIL / segment pred label to profile field key. */
export function personalFieldKeyFromPred(pred: string): string {
  const match = pred.match(/^B-(.+)$/i);
  if (!match) return pred.toLowerCase();
  const label = match[1].toLowerCase();
  if (label === 'other_link') return 'link';
  return label;
}

export function cleanPersonalSegmentText(pred: string, text: string): string {
  return cleanPersonalDisplayValue(personalFieldKeyFromPred(pred), text);
}
