'use client';

import { useMemo, useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { CompareRow, CompareSummary } from '../lib/artifactCompare';

interface Props {
  rows: CompareRow[];
  summary: CompareSummary;
  title: string;
  gtTitle: string;
  trainingPipeline?: string;
  reportFile?: string;
}

function pct(n: number): string {
  return `${n.toFixed(1)}%`;
}

export default function ArtifactCompareTable({
  rows,
  summary,
  title,
  gtTitle,
  trainingPipeline,
  reportFile,
}: Props) {
  const [mismatchesOnly, setMismatchesOnly] = useState(false);

  const visible = useMemo(
    () => (mismatchesOnly ? rows.filter((r) => !r.match && r.groundTruth !== '—') : rows),
    [rows, mismatchesOnly],
  );

  const mismatchCount = rows.filter((r) => !r.match && r.groundTruth !== '—').length;

  return (
    <div className="space-y-3 my-2">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-3">
          <div className="text-[10px] font-mono font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
            Inference
          </div>
          <div className="text-sm font-medium mt-1">{title}</div>
          <div className="text-xs text-[var(--text-secondary)] mt-1">
            {rows.length} tokens in artifact
          </div>
        </div>

        <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] p-3">
          <div className="text-[10px] font-mono font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
            Ground Truth
          </div>
          <div className="text-sm font-medium mt-1">{gtTitle}</div>
          <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span
              className={`text-lg font-semibold tabular-nums ${
                summary.pct >= 90
                  ? 'text-emerald-400'
                  : summary.pct >= 70
                    ? 'text-amber-400'
                    : 'text-red-400'
              }`}
            >
              {summary.matched}/{summary.total}
            </span>
            <span className="text-sm text-[var(--text-secondary)]">
              tokens match ({pct(summary.pct)})
            </span>
          </div>
          {summary.trainingMatched != null && summary.trainingTotal != null && (
            <div className="text-xs text-[var(--text-muted)] mt-1">
              Training checkpoint: {summary.trainingMatched}/{summary.trainingTotal} (
              {pct(summary.trainingPct ?? 0)})
            </div>
          )}
          {trainingPipeline && (
            <div className="text-[10px] font-mono text-[var(--text-muted)] mt-1 truncate">
              {trainingPipeline}
              {reportFile ? ` · ${reportFile}` : ''}
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5 cursor-pointer text-[var(--text-secondary)]">
          <input
            type="checkbox"
            checked={mismatchesOnly}
            onChange={(e) => setMismatchesOnly(e.target.checked)}
            className="rounded"
          />
          Mismatches only ({mismatchCount})
        </label>
        <span className="text-[var(--text-muted)]">Source: {summary.gtSource}</span>
      </div>

      <div className="rounded-lg border border-[var(--border)] overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-16">Match</TableHead>
              <TableHead className="w-28">P/L/T</TableHead>
              <TableHead>Token</TableHead>
              <TableHead className="w-24">Inference</TableHead>
              <TableHead className="w-24">Ground Truth</TableHead>
              {rows.some((r) => r.bioLabel) && <TableHead className="w-24">Mongo Bio</TableHead>}
            </TableRow>
          </TableHeader>
          <TableBody>
            {visible.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="text-center text-[var(--text-muted)] py-6">
                  No rows to display
                </TableCell>
              </TableRow>
            ) : (
              visible.map((row) => (
                <TableRow
                  key={row.key}
                  className={
                    row.groundTruth === '—'
                      ? 'opacity-50'
                      : row.match
                        ? undefined
                        : 'bg-red-500/5'
                  }
                >
                  <TableCell className="text-xs">
                    {row.groundTruth === '—' ? '—' : row.match ? '✅' : '❌'}
                  </TableCell>
                  <TableCell className="font-mono text-[10px]">{row.pageLineToken}</TableCell>
                  <TableCell className="text-xs max-w-[200px] truncate" title={row.token}>
                    {row.token}
                  </TableCell>
                  <TableCell className="font-mono text-[10px]">{row.inference}</TableCell>
                  <TableCell className="font-mono text-[10px]">{row.groundTruth}</TableCell>
                  {rows.some((r) => r.bioLabel) && (
                    <TableCell className="font-mono text-[10px] text-[var(--text-muted)]">
                      {row.bioLabel ?? '—'}
                    </TableCell>
                  )}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {visible.length < rows.length && (
        <p className="text-[10px] text-[var(--text-muted)]">
          Showing {visible.length} of {rows.length} rows
        </p>
      )}
    </div>
  );
}
