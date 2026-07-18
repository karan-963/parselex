/** Display labels for inference-v2 pipeline artifacts (mirrors training-engine/inference_v2/config.py). */

export interface ArtifactCatalogEntry {
  infStep: number;
  trainingPhase: number | null;
  task: string;
  shortLabel: string;
  trainingFolder: string;
}

export const ARTIFACT_CATALOG: Record<string, ArtifactCatalogEntry> = {
  '1_extracted_tokens.json': {
    infStep: 1,
    trainingPhase: null,
    task: 'PDF token extraction',
    shortLabel: 'Extracted tokens',
    trainingFolder: 'extract.py',
  },
  '2_section_headings.json': {
    infStep: 2,
    trainingPhase: 1,
    task: 'Heading detection',
    shortLabel: 'Section headings',
    trainingFolder: 'section/phase1',
  },
  '3_section_labels.json': {
    infStep: 3,
    trainingPhase: 2,
    task: 'Section assignment',
    shortLabel: 'Section labels',
    trainingFolder: 'section/phase2',
  },
  '4_education_segments.json': {
    infStep: 4,
    trainingPhase: 1,
    task: 'Token segmentation',
    shortLabel: 'Education phrase segments',
    trainingFolder: 'education/new_phase1_token_segmentation',
  },
  '5_education_boundaries.json': {
    infStep: 5,
    trainingPhase: 2,
    task: 'Section divider',
    shortLabel: 'Education entry boundaries',
    trainingFolder: 'education/new_phase2_section_divider',
  },
  '6_education_fields.json': {
    infStep: 6,
    trainingPhase: 3,
    task: 'Segment classification',
    shortLabel: 'Education classification',
    trainingFolder: 'education/new_phase3_segment_classification',
  },
  '7_skills_fields.json': {
    infStep: 7,
    trainingPhase: null,
    task: 'Direct classification',
    shortLabel: 'Skills classification',
    trainingFolder: 'skills',
  },
  '8_experience_segments.json': {
    infStep: 8,
    trainingPhase: 1,
    task: 'Token segmentation',
    shortLabel: 'Experience phrase segments',
    trainingFolder: 'experience/phase1_token_segmentation',
  },
  '9_experience_boundaries.json': {
    infStep: 9,
    trainingPhase: 2,
    task: 'Section divider',
    shortLabel: 'Experience entry boundaries',
    trainingFolder: 'experience/phase2_section_divider',
  },
  '10_experience_classification.json': {
    infStep: 10,
    trainingPhase: 3,
    task: 'Segment classification',
    shortLabel: 'Experience classification',
    trainingFolder: 'experience/phase3_segment_classification',
  },
  '11_project_segments.json': {
    infStep: 11,
    trainingPhase: 1,
    task: 'Token segmentation',
    shortLabel: 'Project phrase segments',
    trainingFolder: 'project/phase1_token_segmentation',
  },
  '12_project_boundaries.json': {
    infStep: 12,
    trainingPhase: 2,
    task: 'Section divider',
    shortLabel: 'Project entry boundaries',
    trainingFolder: 'project/phase2_section_divider',
  },
  '13_project_fields.json': {
    infStep: 13,
    trainingPhase: 3,
    task: 'Segment classification',
    shortLabel: 'Project classification',
    trainingFolder: 'project/phase3_segment_classification',
  },
  '14_final_classified_tokens.json': {
    infStep: 14,
    trainingPhase: null,
    task: 'Final merge',
    shortLabel: 'Final classified tokens',
    trainingFolder: 'entities.py',
  },
  '15_personal_fields.json': {
    infStep: 15,
    trainingPhase: null,
    task: 'Segment classification',
    shortLabel: 'Personal classification',
    trainingFolder: 'personal',
  },
  'structured.json': {
    infStep: 14,
    trainingPhase: null,
    task: 'Structured entities',
    shortLabel: 'Structured resume',
    trainingFolder: 'entities.py',
  },
  'performance.json': {
    infStep: 0,
    trainingPhase: null,
    task: 'Pipeline timing and memory',
    shortLabel: 'Performance stats',
    trainingFolder: 'inference_v2/performance.py',
  },
};

/** Pre-swap artifact filenames (older inference runs). */
const LEGACY_ARTIFACT_ALIASES: Record<string, string> = {
  '8_experience_boundaries.json': '9_experience_boundaries.json',
  '9_experience_segments.json': '8_experience_segments.json',
  '10_experience_fields.json': '10_experience_classification.json',
};

export function resolveArtifactCatalogKey(filename: string): string {
  return LEGACY_ARTIFACT_ALIASES[filename] ?? filename;
}

export function formatArtifactAccordionLabel(filename: string): string {
  const entry = ARTIFACT_CATALOG[resolveArtifactCatalogKey(filename)];
  if (!entry) return filename;
  const phase = entry.trainingPhase != null ? ` · training phase ${entry.trainingPhase}` : '';
  return `Step ${entry.infStep} — ${entry.shortLabel}${phase}`;
}

export function formatArtifactAccordionHint(filename: string): string | null {
  const entry = ARTIFACT_CATALOG[resolveArtifactCatalogKey(filename)];
  if (!entry) return null;
  return `${entry.task} · ${entry.trainingFolder}`;
}
