import { resolveArtifactCatalogKey } from './artifactCatalog';

export interface ArtifactTableSpec {
  /** Dot path to array, e.g. `tokens` or `entryDividerLines.lineRows` */
  path: string;
  title: string;
  preferredColumns?: string[];
}

export interface ArtifactPreviewConfig {
  tables: ArtifactTableSpec[];
}

const TOKEN_COLUMNS = [
  'page',
  'lineIndex',
  'tokenIndex',
  'token',
  'prediction',
  'section',
  'bioLabel',
  'x0',
  'y0',
  'x1',
  'y1',
];

// Segmentation tokens surface the model's per-token confidence next to prediction.
const SEG_TOKEN_COLUMNS = [
  'page',
  'lineIndex',
  'tokenIndex',
  'token',
  'prediction',
  'confidence',
  'x0',
  'y0',
  'x1',
  'y1',
];

const CLASSIFY_TOKEN_COLUMNS = [
  'page',
  'lineIndex',
  'tokenIndex',
  'token',
  'prediction',
  'confidence',
  'x0',
  'y0',
  'x1',
  'y1',
];

const BLOCK_ROW_COLUMNS = ['status', 'entryKey', 'gt', 'pred', 'confidence', 'text'];

const ARTIFACT_PREVIEW_CONFIG: Record<string, ArtifactPreviewConfig> = {
  '1_extracted_tokens.json': {
    tables: [{ path: 'tokens', title: 'Tokens', preferredColumns: TOKEN_COLUMNS }],
  },
  '2_section_headings.json': {
    tables: [
      {
        path: 'headings',
        title: 'Headings',
        preferredColumns: ['page', 'lineIndex', 'text', 'source', 'confidence', 'y0'],
      },
    ],
  },
  '3_section_labels.json': {
    tables: [
      {
        path: 'chunks',
        title: 'Section Chunks',
        preferredColumns: ['heading', 'section', 'final_prediction', 'prediction', 'confidence', 'tokenCount'],
      },
    ],
  },
  '4_education_segments.json': {
    tables: [{ path: 'tokens', title: 'Tokens', preferredColumns: SEG_TOKEN_COLUMNS }],
  },
  '5_education_boundaries.json': {
    tables: [
      { path: 'tokens', title: 'Tokens', preferredColumns: CLASSIFY_TOKEN_COLUMNS },
      {
        path: 'entryDividerLines.lineRows',
        title: 'Entry Divider Lines',
        preferredColumns: ['status', 'page', 'line', 'gt', 'pred', 'text'],
      },
    ],
  },
  '6_education_fields.json': {
    tables: [
      { path: 'tokens', title: 'Tokens', preferredColumns: CLASSIFY_TOKEN_COLUMNS },
      {
        path: 'blockClassification.blockRows',
        title: 'Segment Blocks',
        preferredColumns: BLOCK_ROW_COLUMNS,
      },
    ],
  },
  '7_skills_fields.json': {
    tables: [{ path: 'tokens', title: 'Tokens', preferredColumns: CLASSIFY_TOKEN_COLUMNS }],
  },
  '8_experience_segments.json': {
    tables: [{ path: 'tokens', title: 'Tokens', preferredColumns: SEG_TOKEN_COLUMNS }],
  },
  '9_experience_boundaries.json': {
    tables: [{ path: 'tokens', title: 'Tokens', preferredColumns: [...CLASSIFY_TOKEN_COLUMNS, 'isDateToken'] }],
  },
  '10_experience_classification.json': {
    tables: [
      { path: 'tokens', title: 'Tokens', preferredColumns: CLASSIFY_TOKEN_COLUMNS },
      {
        path: 'blockClassification.blockRows',
        title: 'Block Rows',
        preferredColumns: BLOCK_ROW_COLUMNS,
      },
    ],
  },
  '11_project_segments.json': {
    tables: [{ path: 'tokens', title: 'Tokens', preferredColumns: SEG_TOKEN_COLUMNS }],
  },
  '12_project_boundaries.json': {
    tables: [{ path: 'tokens', title: 'Tokens', preferredColumns: CLASSIFY_TOKEN_COLUMNS }],
  },
  '13_project_fields.json': {
    tables: [
      { path: 'tokens', title: 'Tokens', preferredColumns: CLASSIFY_TOKEN_COLUMNS },
      {
        path: 'blockClassification.blockRows',
        title: 'Segment Blocks',
        preferredColumns: BLOCK_ROW_COLUMNS,
      },
    ],
  },
  '14_final_classified_tokens.json': {
    tables: [{ path: 'tokens', title: 'Tokens', preferredColumns: TOKEN_COLUMNS }],
  },
  '15_personal_fields.json': {
    tables: [
      { path: 'tokens', title: 'Tokens', preferredColumns: CLASSIFY_TOKEN_COLUMNS },
      {
        path: 'blockClassification.blockRows',
        title: 'Personal Segments',
        preferredColumns: BLOCK_ROW_COLUMNS,
      },
    ],
  },
};

export function getArtifactPreviewConfig(filename: string): ArtifactPreviewConfig | null {
  return ARTIFACT_PREVIEW_CONFIG[filename] ?? ARTIFACT_PREVIEW_CONFIG[resolveArtifactCatalogKey(filename)] ?? null;
}

export function getValueAtPath(data: unknown, path: string): unknown {
  if (!data || typeof data !== 'object') return undefined;
  return path.split('.').reduce<unknown>((acc, key) => {
    if (!acc || typeof acc !== 'object') return undefined;
    return (acc as Record<string, unknown>)[key];
  }, data);
}

/** Top-level keys consumed entirely by table specs (for metadata panel). */
export function getExcludedTopLevelKeys(config: ArtifactPreviewConfig): Set<string> {
  const keys = new Set<string>();
  for (const table of config.tables) {
    keys.add(table.path.split('.')[0]);
  }
  return keys;
}
