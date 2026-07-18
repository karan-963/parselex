import type { ResumeSection } from './structureResume';

/** Bullet glyphs stripped from anywhere in a value. */
const BULLET_CHARS = /[\u25CF\u2022\u25AA\u25E6\u2023\u2043\u2219\u00B7\u2756]/g;

/** Leading/trailing separator glyphs: hyphen, en/em dash, pipe, plus surrounding space. */
const LEADING_SEPARATORS = /^[\s\-\u2013\u2014|]+/;
const TRAILING_SEPARATORS = /[\s\-\u2013\u2014|]+$/;

/**
 * Repair URLs that were split into multiple tokens during extraction, e.g.
 * `https: //karan963.in/products/resume-tlm` -> `https://karan963.in/products/resume-tlm`.
 * Only spaces that break a scheme (`http(s)://` / `www.`) or its path segments are
 * removed, so ordinary prose is left untouched.
 */
export function stitchUrls(text: string): string {
  // 1. Collapse spaces inside the scheme itself (`https : / /` -> `https://`).
  let out = text.replace(/\bhttps?\s*:\s*\/\s*\/\s*/gi, (match) =>
    match.toLowerCase().includes('https') ? 'https://' : 'http://',
  );
  // 2. Join path segments split with spaces after a URL head (`.com /a /b`).
  out = out.replace(/((?:https?:\/\/|www\.)\S+(?:\s+\/\S*)*)/gi, (match) =>
    match.replace(/\s+/g, ''),
  );
  return out;
}

/**
 * Clean a display value: strip all bullet points, collapse whitespace, repair
 * split URLs, and remove separator characters (`-`, `|`, `—`, `–`) from the
 * start and end. Trailing dots are preserved.
 */
export function cleanDisplayText(text: string): string {
  if (!text) return text;
  let out = text.replace(BULLET_CHARS, ' ');
  out = out.replace(/\s+/g, ' ').trim();
  out = stitchUrls(out);
  out = out.replace(LEADING_SEPARATORS, '');
  out = out.replace(TRAILING_SEPARATORS, '');
  return out.trim();
}

type Fields = { key: string; value: string }[];

const cleanFields = (fields: Fields): Fields =>
  fields
    .map((f) => ({ key: f.key, value: cleanDisplayText(f.value) }))
    .filter((f) => f.value.length > 0);

const cleanItems = (items: string[]): string[] =>
  items.map((item) => cleanDisplayText(item)).filter((item) => item.length > 0);

/** Apply {@link cleanDisplayText} to every rendered value across all sections. */
export function sanitizeResumeSections(sections: ResumeSection[]): ResumeSection[] {
  return sections.map((section) => {
    switch (section.type) {
      case 'profile':
        return { ...section, fields: cleanFields(section.fields) };
      case 'experience':
        return {
          ...section,
          jobs: section.jobs.map((job) => ({ fields: cleanFields(job.fields) })),
        };
      case 'education':
      case 'projects':
        return {
          ...section,
          entries: section.entries.map((entry) => ({ fields: cleanFields(entry.fields) })),
        };
      case 'skills':
      case 'headings':
        return { ...section, items: cleanItems(section.items) };
      default:
        return section;
    }
  });
}
