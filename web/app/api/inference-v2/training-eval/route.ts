import { NextRequest, NextResponse } from 'next/server';
import { readdir, readFile } from 'fs/promises';
import path from 'path';
import {
  getTrainingReportDir,
  getTrainingReportKind,
  parseTrainingReportMd,
} from '@/app/inference-v2/lib/parseTrainingReport';

const REPO_ROOT = path.resolve(process.cwd(), '..');

async function findReportByResumeId(reportDir: string, resumeId: string): Promise<string | null> {
  const absDir = path.join(REPO_ROOT, 'training_pipeline', reportDir);
  let files: string[];
  try {
    files = await readdir(absDir);
  } catch {
    return null;
  }

  const needle = `\`${resumeId}\``;
  for (const name of files.filter((f) => f.endsWith('.md'))) {
    const full = path.join(absDir, name);
    const head = await readFile(full, 'utf-8').then((t) => t.slice(0, 800));
    if (head.includes(needle)) return full;
  }
  return null;
}

export async function GET(req: NextRequest) {
  const pipeline = req.nextUrl.searchParams.get('pipeline');
  const resumeId = req.nextUrl.searchParams.get('resumeId');

  if (!pipeline || !resumeId) {
    return NextResponse.json({ error: 'pipeline and resumeId required' }, { status: 400 });
  }

  const reportDir = getTrainingReportDir(pipeline);
  const kind = getTrainingReportKind(pipeline);
  if (!reportDir || !kind) {
    return NextResponse.json({ error: 'Unknown training pipeline' }, { status: 404 });
  }

  const reportPath = await findReportByResumeId(reportDir, resumeId);
  if (!reportPath) {
    return NextResponse.json({ error: 'Training report not found' }, { status: 404 });
  }

  const text = await readFile(reportPath, 'utf-8');
  const parsed = parseTrainingReportMd(text, kind);
  if (!parsed) {
    return NextResponse.json({ error: 'Failed to parse training report' }, { status: 422 });
  }

  return NextResponse.json({
    ...parsed,
    reportFile: path.basename(reportPath),
    pipeline,
    kind,
  });
}
