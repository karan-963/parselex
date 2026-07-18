/** Parse per-resume training eval markdown reports into comparable token rows. */

export interface TrainingEvalToken {
  page: number;
  lineIndex: number;
  tokenIndex: number;
  token: string;
  bioLabel?: string;
  groundTruth: string;
  trainingPred?: string;
  trainingMatch?: boolean;
}

export interface TrainingEvalReport {
  resumeId: string;
  source: string;
  accuracy?: number;
  correct?: number;
  total?: number;
  tokenCount: number;
  tokens: TrainingEvalToken[];
}

const SEG_ROW_RE =
  /^\|\s*([^|]+)\|\s*P(\d+)\s+L(\d+)\s+T(\d+)\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|/;

const SEG_RESUME_RE = /Phase 1 Phrase Segmentation: `([^`]+)`/;
const SEG_ACCURACY_RE = /\*\*Token Accuracy:\*\*\s*`([\d.]+)%`\s*\((\d+)\/(\d+)\s+correct\)/;

const BOUNDARY_RESUME_RE = /(?:Experience Entry Boundary|Project Boundary Diagnostic):\s*`?([^`\n\r]+?)`?(?:\s|$)/;
const BOUNDARY_LINE_RE =
  /^\|\s*([^|]+)\|\s*P(\d+)\s+L(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|/;

const PIPELINE_REPORT_DIRS: Record<string, { subdir: string; kind: 'seg' | 'boundary' }> = {
  'experience/phase1_token_segmentation': {
    subdir: 'experience/phase1_token_segmentation/reports/minilm/per_resume',
    kind: 'seg',
  },
  'experience/phase2_section_divider': {
    subdir: 'experience/phase2_section_divider/reports/per_resume',
    kind: 'boundary',
  },
  'education/new_phase1_token_segmentation': {
    subdir: 'education/new_phase1_token_segmentation/reports/minilm/per_resume',
    kind: 'seg',
  },
  'education/phase1_token_segmentation': {
    subdir: 'education/phase1_token_segmentation/reports/minilm/per_resume',
    kind: 'seg',
  },
  'education/new_phase2_section_divider': {
    subdir: 'education/new_phase2_section_divider/reports/minilm/per_resume',
    kind: 'boundary',
  },
  'project/phase1_token_segmentation': {
    subdir: 'project/phase1_token_segmentation/reports/minilm/per_resume',
    kind: 'seg',
  },
  'project/phase2_section_divider': {
    subdir: 'project/phase2_section_divider/reports/minilm/per_resume_sparse',
    kind: 'boundary',
  },
};

export function getTrainingReportDir(trainingPipeline: string): string | null {
  const entry = PIPELINE_REPORT_DIRS[trainingPipeline];
  return entry ? entry.subdir : null;
}

export function getTrainingReportKind(trainingPipeline: string): 'seg' | 'boundary' | null {
  return PIPELINE_REPORT_DIRS[trainingPipeline]?.kind ?? null;
}

export function parseSegReportMd(text: string): TrainingEvalReport | null {
  const mId = SEG_RESUME_RE.exec(text);
  if (!mId) return null;

  const resumeId = mId[1];
  let accuracy: number | undefined;
  let correct: number | undefined;
  let total: number | undefined;
  const mAcc = SEG_ACCURACY_RE.exec(text);
  if (mAcc) {
    accuracy = parseFloat(mAcc[1]);
    correct = parseInt(mAcc[2], 10);
    total = parseInt(mAcc[3], 10);
  }

  const tokens: TrainingEvalToken[] = [];
  for (const line of text.split('\n')) {
    const m = SEG_ROW_RE.exec(line.trim());
    if (!m) continue;
    tokens.push({
      page: parseInt(m[2], 10),
      lineIndex: parseInt(m[3], 10),
      tokenIndex: parseInt(m[4], 10),
      token: m[5].trim(),
      bioLabel: m[6].trim(),
      groundTruth: m[7].trim(),
      trainingPred: m[8].trim(),
      trainingMatch: m[1].trim().startsWith('✅'),
    });
  }

  if (tokens.length === 0) return null;

  return {
    resumeId,
    source: 'training_pipeline/seg_report',
    accuracy,
    correct,
    total,
    tokenCount: tokens.length,
    tokens,
  };
}

export function parseBoundaryReportMd(text: string): TrainingEvalReport | null {
  const mId = BOUNDARY_RESUME_RE.exec(text);
  if (!mId) return null;

  const resumeId = mId[1].trim();
  const tokens: TrainingEvalToken[] = [];
  const isProj = text.includes('Project Boundary Diagnostic');
  const PROJ_ROW_RE = /^\|\s*`P(\d+)-L(\d+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|/;

  for (const line of text.split('\n')) {
    if (isProj) {
      const m = PROJ_ROW_RE.exec(line.trim());
      if (!m) continue;
      const gt = m[3].trim() === '[START]' ? 'B-PROJ_START' : 'O';
      const pred = m[4].trim() === '[START]' ? 'B-PROJ_START' : 'O';
      const status = m[5].trim();
      tokens.push({
        page: parseInt(m[1], 10),
        lineIndex: parseInt(m[2], 10),
        tokenIndex: 0,
        token: m[6].trim(),
        groundTruth: gt,
        trainingPred: pred,
        trainingMatch: status === '✓' || status === '✅ MATCH',
      });
    } else {
      const m = BOUNDARY_LINE_RE.exec(line.trim());
      if (!m) continue;
      const gt = m[4].trim();
      const pred = m[5].trim();
      if (!gt.includes('ENTRY') && !pred.includes('ENTRY')) continue;

      tokens.push({
        page: parseInt(m[2], 10),
        lineIndex: parseInt(m[3], 10),
        tokenIndex: 0,
        token: `(line ${m[3]})`,
        groundTruth: gt,
        trainingPred: pred,
        trainingMatch: m[1].trim().startsWith('✅'),
      });
    }
  }

  if (tokens.length === 0) return null;

  return {
    resumeId,
    source: 'training_pipeline/boundary_report',
    tokenCount: tokens.length,
    tokens,
  };
}

export function parseTrainingReportMd(
  text: string,
  kind: 'seg' | 'boundary',
): TrainingEvalReport | null {
  return kind === 'seg' ? parseSegReportMd(text) : parseBoundaryReportMd(text);
}
