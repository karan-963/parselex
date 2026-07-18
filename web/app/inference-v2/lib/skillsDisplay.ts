/** Skills span extraction and badge normalization for inference-v2 UI. */

export interface SkillToken {
  page?: number;
  lineIndex?: number;
  tokenIndex?: number;
  token?: string;
  prediction?: string;
  bioLabel?: string;
}

export interface SkillGroup {
  category?: string;
  skills: string[];
}

const SKILL_BIO = new Set(['B-SKILL', 'I-SKILL']);
const SKILL_TYPE_BIO = new Set(['B-SKILL_TYPE', 'I-SKILL_TYPE']);

const predOf = (t: SkillToken): string => t.prediction ?? t.bioLabel ?? 'O';

/** Drop punctuation-only and empty skill strings. */
export const normalizeSkillBadge = (text: string): string | null => {
  const trimmed = text.trim().replace(/\s+/g, ' ');
  if (!trimmed) return null;
  if (/^[^\w+#./-]+$/.test(trimmed)) return null;
  return trimmed;
};

/** Clean a category label (strip trailing separators like `:` / `-`). */
const normalizeCategory = (text: string): string | undefined => {
  const cleaned = text.trim().replace(/\s+/g, ' ').replace(/[\s:;–—-]+$/, '').trim();
  return cleaned || undefined;
};

/** Dedupe and clean a flat skills list from structured entities. */
export const normalizeSkillList = (items: string[]): string[] => {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of items) {
    const skill = normalizeSkillBadge(raw);
    if (!skill) continue;
    const key = skill.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(skill);
  }
  return out;
};

const sortTokens = (tokens: SkillToken[]): SkillToken[] =>
  [...tokens].sort((a, b) => {
    const ap = a.page ?? 0;
    const bp = b.page ?? 0;
    if (ap !== bp) return ap - bp;
    const al = a.lineIndex ?? 0;
    const bl = b.lineIndex ?? 0;
    if (al !== bl) return al - bl;
    return (a.tokenIndex ?? 0) - (b.tokenIndex ?? 0);
  });

/**
 * Group B-SKILL / I-SKILL spans from step-7 tokens into categories (SKILL_TYPE)
 * and individual skill badge strings.
 */
export const extractSkillGroupsFromTokens = (tokens: SkillToken[]): SkillGroup[] => {
  const sorted = sortTokens(tokens);
  const groups: SkillGroup[] = [];
  let currentCategory: string | undefined;
  let currentSkill = '';

  const ensureGroup = (): SkillGroup => {
    const existing = groups.find((g) => g.category === currentCategory);
    if (existing) return existing;
    const group: SkillGroup = { category: currentCategory, skills: [] };
    groups.push(group);
    return group;
  };

  const flushSkill = () => {
    const skill = normalizeSkillBadge(currentSkill);
    currentSkill = '';
    if (!skill) return;
    const group = ensureGroup();
    if (!group.skills.some((s) => s.toLowerCase() === skill.toLowerCase())) {
      group.skills.push(skill);
    }
  };

  let typeBuffer = '';

  const flushType = () => {
    const cat = normalizeCategory(typeBuffer);
    typeBuffer = '';
    if (cat) currentCategory = cat;
  };

  for (const t of sorted) {
    const pred = predOf(t);
    const tok = (t.token ?? '').trim();

    if (SKILL_TYPE_BIO.has(pred)) {
      // A new category header begins — commit the current skill first.
      flushSkill();
      if (pred === 'B-SKILL_TYPE') {
        flushType();
        typeBuffer = tok;
      } else if (tok) {
        typeBuffer = typeBuffer ? `${typeBuffer} ${tok}` : tok;
      }
      continue;
    }

    if (pred === 'B-SKILL') {
      flushSkill();
      // Commit any pending category header (e.g. "AI/ML Domain :") so this skill
      // and its siblings are nested under it, even when they share the same line.
      flushType();
      currentSkill = tok;
    } else if (pred === 'I-SKILL' && tok) {
      flushType();
      currentSkill = currentSkill ? `${currentSkill} ${tok}` : tok;
    } else if (!SKILL_BIO.has(pred)) {
      // Separator / O token: close the running skill and commit a pending header.
      flushSkill();
      flushType();
    }
  }

  flushSkill();
  return groups.filter((g) => g.skills.length > 0);
};

export const flattenSkillGroups = (groups: SkillGroup[]): string[] => {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const group of groups) {
    for (const skill of group.skills) {
      const key = skill.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(skill);
    }
  }
  return out;
};
