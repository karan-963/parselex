export type ResumeSection =
  | { type: 'profile'; fields: { key: string; value: string }[] }
  | { type: 'summary'; text: string }
  | { type: 'experience'; jobs: { fields: { key: string; value: string }[] }[] }
  | { type: 'education'; entries: { fields: { key: string; value: string }[] }[] }
  | { type: 'projects'; entries: { fields: { key: string; value: string }[] }[] }
  | { type: 'skills'; items: string[] }
  | { type: 'headings'; items: string[] };

import { cleanPersonalDisplayValue } from './personalDisplayUtils';
import { normalizeSkillList } from './skillsDisplay';
import { sanitizeResumeSections } from './textClean';

interface RawEntity {
  label: string;
  value: string;
}

interface RawEntities {
  SECTION_HEADINGS?: string[];
  PERSONAL?: RawEntity[];
  SUMMARY?: string;
  EXPERIENCE?: RawEntity[];
  EXPERIENCE_ENTRIES?: RawEntity[][];
  EDUCATION?: RawEntity[];
  EDUCATION_ENTRIES?: RawEntity[][];
  PROJECTS?: RawEntity[];
  PROJECT_ENTRIES?: RawEntity[][];
  SKILLS?: string[];
}

interface RawStructured {
  entities?: RawEntities;
}

function resolveEntities(data: RawStructured | RawEntities | null | undefined): RawEntities {
  if (!data || typeof data !== 'object') return {};
  if ('entities' in data && data.entities && typeof data.entities === 'object') {
    return data.entities;
  }
  return data as RawEntities;
}

function pushPart(parts: Record<string, string[]>, key: string, value: string) {
  const trimmed = value.trim();
  if (!trimmed) return;
  if (!parts[key]) parts[key] = [];
  parts[key].push(trimmed);
}

function mergeParts(
  parts: Record<string, string[]>,
  order: string[],
): { key: string; value: string }[] {
  const result: { key: string; value: string }[] = [];
  for (const key of order) {
    if (key === 'dates') {
      const start = [...(parts.sdate ?? []), ...(parts.date ?? [])].join(' ').trim();
      const end = (parts.edate ?? []).join(' ').trim();
      let value = '';
      if (start && end) value = `${start} - ${end}`;
      else value = start || end;
      if (value) result.push({ key: 'dates', value });
      continue;
    }
    const joined = (parts[key] ?? []).join(' ').trim();
    if (joined) result.push({ key, value: joined });
  }
  return result;
}

const EXP_FIELD_ORDER = ['role', 'comp', 'dates', 'desc'];
const EDU_FIELD_ORDER = ['degree', 'institution', 'dates', 'grade', 'location', 'desc'];
const PROJ_FIELD_ORDER = ['proj', 'comp', 'dates', 'desc'];

/** True when a DEG value starts a new education row (10th, 12th, B.Tech), not a fragment (CSE, )). */
export const isPrimaryEducationDegreeStart = (value: string): boolean => {
  const trimmed = value.trim();
  if (!trimmed || /^[^\w]+$/.test(trimmed)) return false;
  if (/^\d{1,2}(st|nd|rd|th)\b/i.test(trimmed)) return true;
  if (
    /^(B\.?\s*Tech|M\.?\s*Tech|B\.?\s*E\.?|B\.?\s*Sc\.?|M\.?\s*Sc\.?|MBA|MCA|BCA|Ph\.?\s*D\.?)\b/i.test(
      trimmed,
    )
  ) {
    return true;
  }
  return false;
};

const mapEducationEntityKey = (label: string): string => {
  const upper = label.toUpperCase();
  if (upper === 'DEG' || upper === 'DEGREE') return 'degree';
  if (upper === 'INST' || upper === 'INSTITUTION') return 'institution';
  if (upper === 'DATE' || upper === 'SDATE') return 'date';
  if (upper === 'EDATE') return 'edate';
  if (upper === 'GPA' || upper === 'GRADE') return 'grade';
  if (upper === 'DESC') return 'desc';
  if (upper === 'LOC' || upper === 'LOCATION') return 'location';
  return '';
};

const buildEducationEntryFields = (
  entities: Array<{ label: string; value: string }>,
): { fields: { key: string; value: string }[] } => ({
  fields: mergeParts(
    entities.reduce<Record<string, string[]>>((parts, entity) => {
      const key = mapEducationEntityKey(entity.label);
      if (key) pushPart(parts, key, entity.value);
      return parts;
    }, {}),
    EDU_FIELD_ORDER,
  ),
});

const mapExperienceEntityKey = (label: string): string => {
  const upper = label.toUpperCase();
  if (upper === 'ROLE') return 'role';
  if (upper === 'COMP') return 'comp';
  if (upper === 'SDATE' || upper === 'DATE') return 'sdate';
  if (upper === 'EDATE') return 'edate';
  if (upper === 'DESC') return 'desc';
  return '';
};

const buildExperienceEntryFields = (
  entities: Array<{ label: string; value: string }>,
): { fields: { key: string; value: string }[] } => ({
  fields: mergeParts(
    entities.reduce<Record<string, string[]>>((parts, entity) => {
      const key = mapExperienceEntityKey(entity.label);
      if (key) pushPart(parts, key, entity.value);
      return parts;
    }, {}),
    EXP_FIELD_ORDER,
  ),
});

const mapProjectEntityKey = (label: string): string => {
  const upper = label.toUpperCase();
  if (upper === 'PROJ_NAME' || upper === 'PROJ') return 'proj';
  if (upper === 'PROJ_COMPANY' || upper === 'COMP') return 'comp';
  if (upper === 'SDATE' || upper === 'DATE') return 'sdate';
  if (upper === 'EDATE') return 'edate';
  if (upper === 'DESC') return 'desc';
  return '';
};

const buildProjectEntryFields = (
  entities: Array<{ label: string; value: string }>,
): { fields: { key: string; value: string }[] } => ({
  fields: mergeParts(
    entities.reduce<Record<string, string[]>>((parts, entity) => {
      const key = mapProjectEntityKey(entity.label);
      if (key) pushPart(parts, key, entity.value);
      return parts;
    }, {}),
    PROJ_FIELD_ORDER,
  ),
});

export function structureResume(data: RawStructured | RawEntities | null | undefined): ResumeSection[] {
  const sections: ResumeSection[] = [];
  const entities = resolveEntities(data);

  // 1. Profile Section
  if (entities.PERSONAL && Array.isArray(entities.PERSONAL)) {
    const profileMap: Record<string, string[]> = {};
    for (const entity of entities.PERSONAL) {
      if (!entity.label || !entity.value) continue;
      const label = entity.label.toUpperCase();
      let key = '';

      if (label === 'NAME') key = 'name';
      else if (label === 'EMAIL') key = 'email';
      else if (label === 'PHONE') key = 'phone';
      else if (label === 'GITHUB') key = 'github';
      else if (label === 'LINKEDIN') key = 'linkedin';
      else if (label === 'OTHER_LINK') key = 'link';
      else if (label === 'LOCATION') key = 'location';

      if (key) {
        if (!profileMap[key]) {
          profileMap[key] = [];
        }
        profileMap[key].push(cleanPersonalDisplayValue(key, entity.value));
      }
    }

    const order = ['name', 'email', 'phone', 'github', 'linkedin', 'link', 'location'];
    // Multi-valued keys (a resume can have >1 GitHub/LinkedIn/other link) get
    // numbered rows instead of being joined into one mashed string.
    const fields = order
      .filter(k => profileMap[k] && profileMap[k].length > 0)
      .flatMap(k => {
        const values = profileMap[k];
        if (values.length === 1) return [{ key: k, value: values[0] }];
        return values.map((value, i) => ({ key: `${k} ${i + 1}`, value }));
      });

    if (fields.length > 0) {
      sections.push({ type: 'profile', fields });
    }
  }

  // 1b. Summary — pure heuristic (whole SUMMARY-section text), no model involved.
  if (entities.SUMMARY && entities.SUMMARY.trim()) {
    sections.push({ type: 'summary', text: entities.SUMMARY.trim() });
  }

  // 2. Experience Section — group by step-9 entry boundaries (EXPERIENCE_ENTRIES)
  // when available, falling back to the primary-role heuristic for legacy runs.
  if (entities.EXPERIENCE_ENTRIES?.length) {
    const jobs = entities.EXPERIENCE_ENTRIES
      .map((entry) => buildExperienceEntryFields(entry))
      .filter((job) => job.fields.length > 0);
    if (jobs.length > 0) {
      sections.push({ type: 'experience', jobs });
    }
  } else if (entities.EXPERIENCE && Array.isArray(entities.EXPERIENCE)) {
    const jobs: { fields: { key: string; value: string }[] }[] = [];
    let currentParts: Record<string, string[]> = {};
    let hasRole = false;

    const flushJob = () => {
      const fields = mergeParts(currentParts, EXP_FIELD_ORDER);
      if (fields.length > 0) jobs.push({ fields });
      currentParts = {};
      hasRole = false;
    };

    for (const entity of entities.EXPERIENCE) {
      if (!entity.label || !entity.value) continue;
      const label = entity.label.toUpperCase();
      let key = '';

      if (label === 'ROLE') key = 'role';
      else if (label === 'COMP') key = 'comp';
      else if (label === 'SDATE' || label === 'DATE') key = 'sdate';
      else if (label === 'EDATE') key = 'edate';
      else if (label === 'DESC') key = 'desc';

      if (!key) continue;

      const isPrimaryRoleStart = (value: string) => {
        const v = value.trim();
        if (v.startsWith('•') || v.startsWith('●')) return true;
        if (/present\s*\)?$/i.test(v)) return false;
        if (/^\(?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)/i.test(v)) return false;
        return false;
      };

      if (key === 'role' && hasRole && isPrimaryRoleStart(entity.value)) flushJob();
      if (key === 'role') hasRole = true;

      pushPart(currentParts, key, entity.value);
    }

    flushJob();

    if (jobs.length > 0) {
      sections.push({ type: 'experience', jobs });
    }
  }

  // 3. Education Section — group by step-5 entry boundaries (EDUCATION_ENTRIES) or primary degree rows
  const educationEntries = entities.EDUCATION_ENTRIES as Array<Array<{ label: string; value: string }>> | undefined;
  if (educationEntries?.length) {
    const entriesList = educationEntries
      .map((entry) => buildEducationEntryFields(entry))
      .filter((entry) => entry.fields.length > 0);
    if (entriesList.length > 0) {
      sections.push({ type: 'education', entries: entriesList });
    }
  } else if (entities.EDUCATION && Array.isArray(entities.EDUCATION)) {
    const entriesList: { fields: { key: string; value: string }[] }[] = [];
    let currentParts: Record<string, string[]> = {};
    let hasPrimaryDegree = false;

    const flushEdu = () => {
      const fields = mergeParts(currentParts, EDU_FIELD_ORDER);
      if (fields.length > 0) entriesList.push({ fields });
      currentParts = {};
      hasPrimaryDegree = false;
    };

    for (const entity of entities.EDUCATION) {
      if (!entity.label || !entity.value) continue;
      const key = mapEducationEntityKey(entity.label);
      if (!key) continue;

      if (key === 'degree' && hasPrimaryDegree && isPrimaryEducationDegreeStart(entity.value)) {
        flushEdu();
      }
      if (key === 'degree' && isPrimaryEducationDegreeStart(entity.value)) {
        hasPrimaryDegree = true;
      } else if (key === 'degree') {
        // Continuation fragment (CSE, )) — merge into current entry degree field
        pushPart(currentParts, key, entity.value);
        continue;
      }

      pushPart(currentParts, key, entity.value);
    }

    flushEdu();

    if (entriesList.length > 0) {
      sections.push({ type: 'education', entries: entriesList });
    }
  }

  // 4. Projects Section — group by step-12 entry boundaries (PROJECT_ENTRIES)
  // when available, falling back to PROJ_NAME flush heuristic for legacy runs.
  if (entities.PROJECT_ENTRIES?.length) {
    const projEntries = entities.PROJECT_ENTRIES
      .map((entry) => buildProjectEntryFields(entry))
      .filter((entry) => entry.fields.length > 0);
    if (projEntries.length > 0) {
      sections.push({ type: 'projects', entries: projEntries });
    }
  } else if (entities.PROJECTS && Array.isArray(entities.PROJECTS)) {
    const projEntries: { fields: { key: string; value: string }[] }[] = [];
    let currentParts: Record<string, string[]> = {};
    let hasProj = false;

    const flushProj = () => {
      const fields = mergeParts(currentParts, PROJ_FIELD_ORDER);
      if (fields.length > 0) projEntries.push({ fields });
      currentParts = {};
      hasProj = false;
    };

    for (const entity of entities.PROJECTS) {
      if (!entity.label || !entity.value) continue;
      const key = mapProjectEntityKey(entity.label);
      if (!key) continue;

      if (key === 'proj' && hasProj) flushProj();
      if (key === 'proj') hasProj = true;

      pushPart(currentParts, key, entity.value);
    }

    flushProj();

    if (projEntries.length > 0) {
      sections.push({ type: 'projects', entries: projEntries });
    }
  }

  // 5. Skills Section
  if (entities.SKILLS && Array.isArray(entities.SKILLS) && entities.SKILLS.length > 0) {
    const cleaned = normalizeSkillList(entities.SKILLS);
    if (cleaned.length > 0) {
      sections.push({ type: 'skills', items: cleaned });
    }
  }

  // 6. Headings Section
  if (entities.SECTION_HEADINGS && Array.isArray(entities.SECTION_HEADINGS) && entities.SECTION_HEADINGS.length > 0) {
    sections.push({ type: 'headings', items: entities.SECTION_HEADINGS });
  }

  return sanitizeResumeSections(sections);
}
