'use client';

import { useEffect, useMemo, useState } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import { buildEntryDividerReport, buildPredictedEntryReport, EntryDividerLineRow } from '../lib/entryDividerLines';
import { isGtEnabled, GT_ENABLED } from '../lib/gtGate';

interface Props {
  slug: string;
  resumeId?: string;
  filename?: string;
  artifact: {
    title?: string;
    section?: string;
    entryDividerLines?: ReturnType<typeof buildEntryDividerReport>;
    tokens?: { page: number; lineIndex: number; prediction?: string }[];
    tokenCount?: number;
  };
}

function LineTable({ rows, showGt = GT_ENABLED }: { rows: EntryDividerLineRow[]; showGt?: boolean }) {
  const colCount = showGt ? 7 : 6;
  return (
    <div className="overflow-x-auto border border-[var(--border)] rounded-lg">
      <table className="w-full text-left border-collapse text-xs">
        <thead className="bg-[var(--bg-elevated)] border-b border-[var(--border)] font-mono text-[10px] uppercase tracking-wider text-[var(--text-secondary)]">
          <tr>
            <th className="p-2.5 font-semibold">Status</th>
            <th className="p-2.5 font-semibold">P/L</th>
            {showGt && <th className="p-2.5 font-semibold">GT</th>}
            <th className="p-2.5 font-semibold">Pred</th>
            <th className="p-2.5 font-semibold">Conf</th>
            <th className="p-2.5 font-semibold">Token labels</th>
            <th className="p-2.5 font-semibold">Line text</th>
          </tr>
        </thead>
        <tbody className="font-mono text-[11px]">
          {rows.length === 0 ? (
            <tr>
              <td colSpan={colCount} className="p-3 text-[var(--text-secondary)]">No entry divider lines.</td>
            </tr>
          ) : (
            rows.map((row) => (
              <tr key={`${row.page}-${row.line}`} className="border-b border-[var(--border)]/50 hover:bg-[var(--bg-elevated)]/50">
                <td className="p-2.5">{row.status}</td>
                <td className="p-2.5 whitespace-nowrap">P{row.page} L{row.line}</td>
                {showGt && <td className="p-2.5">{row.gt || '—'}</td>}
                <td className="p-2.5">{row.pred || '—'}</td>
                <td className={`p-2.5 whitespace-nowrap ${row.confidence != null && row.confidence < 0.5 ? 'text-amber-500' : ''}`}>
                  {row.confidence != null ? `${(row.confidence * 100).toFixed(1)}%` : '—'}
                </td>
                <td className="p-2.5 text-[var(--text-secondary)]">{row.tokenLabels}</td>
                <td className="p-2.5 text-[var(--text-primary)] max-w-lg truncate" title={row.text}>{row.text}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export default function EntrySectionDividerView({ slug, resumeId, filename, artifact }: Props) {
  const [dbResumeData, setDbResumeData] = useState<any>(null);
  const [boundaryArtifact, setBoundaryArtifact] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [gtRefreshTick, setGtRefreshTick] = useState(0);

  const isEducation =
    filename?.includes('education_boundaries') ||
    artifact.section === 'EDUCATION';
  const isProject =
    !isEducation && (
      filename?.includes('project_boundaries') ||
      artifact.section === 'PROJECT'
    );
  const isBoundaryArtifact = !!filename?.includes('_boundaries.json');
  const boundaryFilename = isEducation
    ? '5_education_boundaries.json'
    : isProject
      ? '12_project_boundaries.json'
      : '9_experience_boundaries.json';
  const sectionKey: 'EXPERIENCE' | 'PROJECT' | 'EDUCATION' = isEducation
    ? 'EDUCATION'
    : isProject
      ? 'PROJECT'
      : 'EXPERIENCE';

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const loads: Promise<void>[] = [];

    if (resumeId && isGtEnabled(resumeId)) {
      loads.push(
        fetch(`/api/resumes/${encodeURIComponent(resumeId)}`, { cache: 'no-store' })
          .then((r) => (r.ok ? r.json() : null))
          .then((data) => {
            if (!cancelled && data) setDbResumeData(data);
          }),
      );
    }

    if (isBoundaryArtifact && artifact.tokens) {
      setBoundaryArtifact({ tokens: artifact.tokens });
    } else {
      loads.push(
        fetch(`/api/inference-v2/runs/${encodeURIComponent(slug)}/artifacts/${encodeURIComponent(boundaryFilename)}`)
          .then((r) => (r.ok ? r.json() : null))
          .then((data) => {
            if (!cancelled) setBoundaryArtifact(data);
          }),
      );
    }

    Promise.all(loads)
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [slug, resumeId, gtRefreshTick, isBoundaryArtifact, artifact.tokens, boundaryFilename]);

  const report = useMemo(() => {
    if (!dbResumeData) return null;
    return buildEntryDividerReport(
      dbResumeData,
      boundaryArtifact?.tokens ?? artifact.tokens ?? [],
      artifact.entryDividerLines ?? null,
      sectionKey,
    );
  }, [dbResumeData, boundaryArtifact, artifact.entryDividerLines, artifact.tokens, isProject, isEducation]);

  // Prediction-only view — used whenever there's no MongoDB GT to compare against
  // (uploaded resumes, or a GT-gated resume whose entry heads haven't been labeled yet).
  const predictedReport = useMemo(() => {
    const tokens = boundaryArtifact?.tokens ?? artifact.tokens ?? [];
    if (!tokens.length) return null;
    return buildPredictedEntryReport(tokens, sectionKey);
  }, [boundaryArtifact, artifact.tokens, sectionKey]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)] py-4">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading entry divider comparison…
      </div>
    );
  }

  if (!report) {
    const stepNum = isEducation ? '5' : isProject ? '12' : '9';
    const startLabel = isEducation ? 'B-EDU_START' : isProject ? 'B-PROJ_START' : 'B-ENTRY';
    const entryNoun = isEducation ? 'education' : isProject ? 'project' : 'job';
    const headsKey = isEducation
      ? 'educationEntryHeads'
      : isProject
        ? 'projectEntryHeads'
        : 'experienceEntryHeads';
    const predRows = predictedReport?.lineRows ?? [];
    return (
      <div className="space-y-3 my-2">
        <div className="text-[10px] text-[var(--text-secondary)] space-y-1 max-w-2xl">
          <p>
            <span className="font-semibold text-[var(--text-primary)]">
              Step {stepNum} — Predicted entry boundaries
            </span>
            . Each row is a predicted <strong>{entryNoun} entry start line</strong>{' '}
            (<code>{startLabel}</code>).
            {GT_ENABLED && (
              isGtEnabled(resumeId)
                ? <> No MongoDB <code>{headsKey}</code> found for this resume yet — showing model predictions only. Save entry heads in the viewer, then Refresh GT.</>
                : <> Ground-truth comparison is only available for the default reference resume; this is an uploaded resume, so only model predictions are shown.</>
            )}
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 p-3 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] text-xs max-w-xs">
          <div>
            <div className="text-[10px] text-[var(--text-secondary)] uppercase font-mono">Predicted entries</div>
            <div className="text-base font-bold font-mono text-[var(--text-primary)]">{predRows.length}</div>
          </div>
        </div>
        <LineTable rows={predRows} />
      </div>
    );
  }

  const { metrics } = report;
  const stepNum = isEducation ? '5' : isProject ? '12' : '9';
  const boundaryLabel = isEducation ? 'B-EDU_START' : isProject ? 'B-PROJ_START' : 'B-ENTRY';
  const headsKey = isEducation
    ? 'educationEntryHeads'
    : isProject
      ? 'projectEntryHeads'
      : 'experienceEntryHeads';
  const trainingFolder = isEducation
    ? 'education/new_phase2_section_divider'
    : isProject
      ? 'project/phase2_section_divider'
      : 'experience/phase2_section_divider';
  const boundaryStep = isEducation ? '5' : isProject ? '12' : '9';
  const entryNoun = isEducation ? 'education' : isProject ? 'project' : 'job';

  return (
    <div className="space-y-4 my-2">
      <div className="flex items-start justify-between gap-2">
        <div className="text-[10px] text-[var(--text-secondary)] space-y-1 max-w-2xl">
          <p>
            <span className="font-semibold text-[var(--text-primary)]">
              Step {stepNum} — Entry section divider
            </span>
            {' '}(training: <code>{trainingFolder}</code>).
            Each row is a <strong>{entryNoun} entry start line</strong> from MongoDB{' '}
            <code>{headsKey}</code>, compared to step {boundaryStep}{' '}
            <code>{boundaryLabel}</code> predictions.
          </p>
          {!isBoundaryArtifact && (
            <p>
              The JSON <code>tokens[]</code> in this artifact are internal phrase-segmentation labels (
              <code>B-SEG</code>/<code>I-SEG</code>) for the next step — not the divider GT shown here.
            </p>
          )}
          {isBoundaryArtifact && isProject && (
            <p>
              Per-token <code>B-PROJ_START</code>/<code>I-PROJ_START</code> labels are in the artifact{' '}
              <code>tokens[]</code>; this table is the primary eval (boundary FBA).
            </p>
          )}
          {isBoundaryArtifact && isEducation && (
            <p>
              Per-token <code>B-EDU_START</code> labels are in the artifact{' '}
              <code>tokens[]</code>; this table is the primary eval (boundary FBA).
            </p>
          )}
          {isBoundaryArtifact && !isEducation && !isProject && (
            <p>
              Per-token <code>B-ENTRY</code>/<code>I-ENTRY</code> labels are in the artifact{' '}
              <code>tokens[]</code>; this table is the primary eval (boundary FBA).
            </p>
          )}
        </div>
        {resumeId && (
          <button
            type="button"
            onClick={() => setGtRefreshTick((n) => n + 1)}
            className="inline-flex items-center gap-1 px-2 py-1 rounded border border-[var(--border)] text-[10px] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
          >
            <RefreshCw className="h-3 w-3" />
            Refresh GT
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-6 gap-2 p-3 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] text-xs">
        <div>
          <div className="text-[10px] text-[var(--text-secondary)] uppercase font-mono">FBA</div>
          <div className={`text-base font-bold font-mono ${metrics.fbaPercent >= 90 ? 'text-green-400' : 'text-yellow-400'}`}>
            {metrics.fbaPercent.toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--text-secondary)] uppercase font-mono">Pred entries</div>
          <div className="text-base font-bold font-mono">{report.predEntryLines.length}</div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--text-secondary)] uppercase font-mono">GT lines</div>
          <div className="text-base font-bold font-mono">{metrics.gtEntryLines}</div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--text-secondary)] uppercase font-mono">Matched</div>
          <div className="text-base font-bold font-mono text-green-400">{metrics.matched}</div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--text-secondary)] uppercase font-mono">Missed</div>
          <div className="text-base font-bold font-mono text-red-400">{metrics.missed}</div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--text-secondary)] uppercase font-mono">Extra</div>
          <div className="text-base font-bold font-mono text-orange-400">{metrics.extra}</div>
        </div>
      </div>

      <div>
        <h4 className="text-xs font-semibold text-[var(--text-primary)] mb-1">Entry line comparison</h4>
        <p className="text-[10px] text-[var(--text-secondary)] mb-2">
          GT source: <code>{report.gtSource}</code> — same table as{' '}
          <code>{trainingFolder}/reports/minilm/per_resume_sparse/*.md</code>
        </p>
        <LineTable rows={report.lineRows} />
      </div>
    </div>
  );
}
