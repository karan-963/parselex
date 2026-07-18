'use client';

import { useEffect, useMemo, useState } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import {
  buildFieldClassificationReport,
  buildPredictedFieldReport,
  FieldClassificationRow,
} from '../lib/fieldClassificationReport';
import { cleanPersonalSegmentText } from '../lib/personalDisplayUtils';
import { isGtEnabled, GT_ENABLED } from '../lib/gtGate';

interface Props {
  slug: string;
  resumeId?: string;
  artifact: {
    section?: string;
    blockClassification?: ReturnType<typeof buildFieldClassificationReport> extends infer R
      ? R extends null ? never : R
      : never;
    tokens?: unknown[];
  };
}

function BlockTable({
  rows,
  segmentMode,
  personalMode,
}: {
  rows: FieldClassificationRow[];
  segmentMode?: boolean;
  personalMode?: boolean;
}) {
  return (
    <div className="overflow-x-auto border border-[var(--border)] rounded-lg max-h-[50vh]">
      <table className="w-full text-left border-collapse text-xs">
        <thead className="bg-[var(--bg-elevated)] sticky top-0 border-b border-[var(--border)] font-mono text-[10px] uppercase tracking-wider text-[var(--text-secondary)]">
          <tr>
            <th className="p-2.5 font-semibold">Status</th>
            <th className="p-2.5 font-semibold">{segmentMode ? 'Seg #' : 'Entry'}</th>
            <th className="p-2.5 font-semibold">GT</th>
            <th className="p-2.5 font-semibold">Pred</th>
            <th className="p-2.5 font-semibold">Conf</th>
            <th className="p-2.5 font-semibold">Text</th>
          </tr>
        </thead>
        <tbody className="font-mono text-[11px]">
          {rows.length === 0 ? (
            <tr>
              <td colSpan={6} className="p-3 text-[var(--text-secondary)]">No blocks scored.</td>
            </tr>
          ) : (
            rows.map((row, i) => (
              <tr key={`${row.entryKey}-${i}`} className="border-b border-[var(--border)]/50 hover:bg-[var(--bg-elevated)]/50">
                <td className="p-2.5">{row.status}</td>
                <td className="p-2.5 whitespace-nowrap">{row.entryKey}</td>
                <td className="p-2.5">{row.gt}</td>
                <td className="p-2.5">{row.pred}</td>
                <td className="p-2.5 tabular-nums">
                  {typeof row.confidence === 'number' ? row.confidence.toFixed(4) : '—'}
                </td>
                <td className="p-2.5 text-[var(--text-primary)] max-w-lg truncate" title={row.text}>
                  {personalMode ? cleanPersonalSegmentText(row.pred, row.text) : row.text}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export default function FieldClassificationView({ slug, resumeId, artifact }: Props) {
  const [dbResumeData, setDbResumeData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [gtRefreshTick, setGtRefreshTick] = useState(0);

  useEffect(() => {
    if (!resumeId || !isGtEnabled(resumeId)) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    fetch(`/api/resumes/${encodeURIComponent(resumeId)}`, { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!cancelled && data) setDbResumeData(data);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [resumeId, gtRefreshTick]);

  const report = useMemo(() => {
    if (GT_ENABLED && artifact.blockClassification?.blockRows?.length) {
      return artifact.blockClassification;
    }
    if (!dbResumeData) return null;
    return buildFieldClassificationReport(dbResumeData, artifact);
  }, [artifact, dbResumeData]);

  if (loading && !artifact.blockClassification) {
    return (
      <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)] py-4">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading field classification…
      </div>
    );
  }

  const isProject = artifact.section === 'PROJECT';
  const isEducation = artifact.section === 'EDUCATION';
  const isPersonal = artifact.section === 'PERSONAL';

  if (!report) {
    const predRows = buildPredictedFieldReport(
      Array.isArray(artifact.tokens) ? (artifact.tokens as any[]) : [],
    );
    const stepNum = isEducation ? '6' : isProject ? '13' : isPersonal ? '15' : '10';
    const unit = isPersonal ? 'segment' : isEducation || isProject ? 'segment' : 'block';
    const classList = isEducation
      ? 'INSTITUTION / DEGREE / DATE / DESCRIPTION'
      : isProject
        ? 'PROJECT_NAME / SDATE / EDATE / DESC'
        : isPersonal
          ? 'NAME / PHONE / EMAIL / LOCATION / …'
          : 'ROLE / COMP / DATE / DESC';
    const gtUnavailableNote = !GT_ENABLED
      ? null
      : isGtEnabled(resumeId)
        ? <> No MongoDB labels found for this resume yet — showing model predictions only. Save labels in the viewer, then Refresh GT.</>
        : <> Ground-truth comparison is only available for the default reference resume; this is an uploaded resume, so only model predictions are shown.</>;
    return (
      <div className="space-y-3 my-2">
        <div className="text-[10px] text-[var(--text-secondary)] space-y-1 max-w-2xl">
          <p>
            <span className="font-semibold text-[var(--text-primary)]">
              Step {stepNum} — Predicted field classification
            </span>
            . Each row is a predicted {unit} classified as {classList}.{gtUnavailableNote}
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 p-3 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] text-xs max-w-xs">
          <div>
            <div className="text-[10px] text-[var(--text-secondary)] uppercase font-mono">
              Predicted {unit}s
            </div>
            <div className="text-base font-bold font-mono text-[var(--text-primary)]">
              {predRows.length}
            </div>
          </div>
        </div>
        <div className="overflow-x-auto border border-[var(--border)] rounded-lg max-h-[50vh]">
          <table className="w-full text-left border-collapse text-xs">
            <thead className="bg-[var(--bg-elevated)] sticky top-0 border-b border-[var(--border)] font-mono text-[10px] uppercase tracking-wider text-[var(--text-secondary)]">
              <tr>
                <th className="p-2.5 font-semibold">#</th>
                <th className="p-2.5 font-semibold">P/L</th>
                <th className="p-2.5 font-semibold">Prediction</th>
                <th className="p-2.5 font-semibold">Conf</th>
                <th className="p-2.5 font-semibold">Text</th>
              </tr>
            </thead>
            <tbody className="font-mono text-[11px]">
              {predRows.length === 0 ? (
                <tr>
                  <td colSpan={5} className="p-3 text-[var(--text-secondary)]">
                    No classified tokens in this artifact.
                  </td>
                </tr>
              ) : (
                predRows.map((row) => (
                  <tr
                    key={row.segIndex}
                    className="border-b border-[var(--border)]/50 hover:bg-[var(--bg-elevated)]/50"
                  >
                    <td className="p-2.5 whitespace-nowrap">{row.segIndex}</td>
                    <td className="p-2.5 whitespace-nowrap text-[var(--text-secondary)]">
                      P{row.page} L{row.lineIndex}
                    </td>
                    <td className="p-2.5">
                      <code className="px-1 py-0.5 rounded text-[11px] bg-[var(--accent)]/10 text-[var(--accent)] border border-[var(--accent)]/20">
                        {row.label}
                      </code>
                    </td>
                    <td className="p-2.5 tabular-nums">{row.confidence.toFixed(4)}</td>
                    <td
                      className="p-2.5 text-[var(--text-primary)] max-w-lg truncate"
                      title={row.text}
                    >
                      {isPersonal ? cleanPersonalSegmentText(row.label, row.text) : row.text}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  const { metrics } = report;
  const segmentMode = isProject || isEducation || isPersonal;
  const accuracyLabel = segmentMode ? 'Segment accuracy' : 'Block accuracy';
  const countLabel = segmentMode ? 'Segments' : 'Blocks';
  const tableTitle = segmentMode ? 'Segment predictions' : 'Block predictions';

  return (
    <div className="space-y-4 my-2">
      <div className="flex items-start justify-between gap-2">
        <div className="text-[10px] text-[var(--text-secondary)] space-y-1 max-w-2xl">
          {isEducation ? (
            <>
              <p>
                <span className="font-semibold text-[var(--text-primary)]">Step 6 — Field classification</span>
                {' '}(training: <code>education/new_phase3_segment_classification</code>).
                Each row is a <strong>phrase segment</strong> classified as INSTITUTION / DEGREE / DATE / DESCRIPTION.
              </p>
              <p>
                Segments are built with <code>construct_sentences_by_appearance</code> on EDUCATION tokens.
                GT uses MongoDB <code>bioLabel</code> segment majority voting.
              </p>
            </>
          ) : isPersonal ? (
            <>
              <p>
                <span className="font-semibold text-[var(--text-primary)]">Step 15 — Field classification</span>
                {' '}(training: <code>personal</code>).
                Each row is an <strong>atomic personal segment</strong> classified as NAME / PHONE / EMAIL / LOCATION / …
              </p>
              <p>
                Segments are built with <code>build_personal_segments</code> on PERSONAL tokens.
                GT uses MongoDB <code>bioLabel</code> via <code>derive_segment_label</code> (B vs I equivalence).
              </p>
            </>
          ) : isProject ? (
            <>
              <p>
                <span className="font-semibold text-[var(--text-primary)]">Step 13 — Field classification</span>
                {' '}(training: <code>project/phase3_segment_classification</code>).
                Each row is a <strong>phrase segment</strong> classified as PROJECT_NAME / SDATE / EDATE / DESC.
              </p>
              <p>
                Segments are built with <code>construct_sentences_by_appearance</code> on PROJECT tokens.
                GT uses MongoDB <code>bioLabel</code> segment majority voting + DATE resolution.
              </p>
            </>
          ) : (
            <>
              <p>
                <span className="font-semibold text-[var(--text-primary)]">Step 10 — Field classification</span>
                {' '}(training: <code>phase3_segment_classification</code>).
                Each row is a <strong>phrase block</strong> classified as ROLE / COMP / DATE / DESC.
              </p>
              <p>
                Job entries are split using step 8 boundaries + step 9 <code>B-SEG</code> dividers.
                GT uses MongoDB <code>bioLabel</code> (first <code>B-*</code> per block).
                Entry heads from <code>experienceEntryHeads</code>.
              </p>
            </>
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

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 p-3 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] text-xs">
        <div>
          <div className="text-[10px] text-[var(--text-secondary)] uppercase font-mono">{accuracyLabel}</div>
          <div className={`text-base font-bold font-mono ${metrics.macroF1ProxyPercent >= 90 ? 'text-green-400' : 'text-yellow-400'}`}>
            {metrics.macroF1ProxyPercent.toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--text-secondary)] uppercase font-mono">{countLabel}</div>
          <div className="text-base font-bold font-mono">{metrics.blocks}</div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--text-secondary)] uppercase font-mono">Correct</div>
          <div className="text-base font-bold font-mono text-green-400">{metrics.correct}</div>
        </div>
        <div>
          <div className="text-[10px] text-[var(--text-secondary)] uppercase font-mono">Errors</div>
          <div className="text-base font-bold font-mono text-red-400">{metrics.errors}</div>
        </div>
      </div>

      <div>
        <h4 className="text-xs font-semibold text-[var(--text-primary)] mb-1">{tableTitle}</h4>
        <p className="text-[10px] text-[var(--text-secondary)] mb-2">
          Same format as{' '}
          <code>
            {isEducation
              ? 'education/new_phase3_segment_classification/reports/minilm/per_resume/*.md'
              : isProject
                ? 'project/phase3_segment_classification/reports/minilm/per_resume/*.md'
                : isPersonal
                  ? 'personal/reports/minilm/per_resume/*.md'
                  : 'phase3_segment_classification/reports/minilm/per_resume/*.md'}
          </code>
        </p>
        <BlockTable rows={report.blockRows} segmentMode={segmentMode} personalMode={isPersonal} />
      </div>
    </div>
  );
}
