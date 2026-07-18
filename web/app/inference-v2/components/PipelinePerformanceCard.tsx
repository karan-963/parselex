import SectionCard from './SectionCard';
import PerformanceRadialPanel from './PerformanceRadialPanel';
import {
  formatDurationMs,
  formatMemoryMb,
  type PipelinePerformanceStats,
  wallClockDurationMs,
} from '../lib/formatPerformance';
import { formatPercent, isVisibleSection } from '../lib/sectionPerformance';
import { GT_ENABLED } from '../lib/gtGate';

interface Props {
  stats?: PipelinePerformanceStats | null;
  createdAt?: string;
  completedAt?: string | null;
}

const SECTION_ORDER = [
  'section_headings',
  'profile',
  'education',
  'skills',
  'experience',
  'projects',
] as const;

export default function PipelinePerformanceCard({ stats, createdAt, completedAt }: Props) {
  const fallbackMs = wallClockDurationMs(createdAt, completedAt);
  const totalMs = stats?.totalDurationMs ?? fallbackMs;
  const stages = stats?.stages ?? [];
  const peakMem = stats?.peakMemoryMb;
  const sections = stats?.sections ?? {};
  const sectionRows = SECTION_ORDER.map((key) => sections[key]).filter(isVisibleSection);

  if (totalMs == null && stages.length === 0) {
    return null;
  }

  return (
    <SectionCard title="Performance">
      {sectionRows.length > 0 ? (
        <div className="space-y-5">
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_220px] gap-5 items-start">
            <div>
              <h4 className="text-xs font-semibold uppercase tracking-wide text-[var(--text-secondary)] mb-2">
                By section
              </h4>
              <div className="overflow-x-auto rounded-lg border border-[var(--border)]">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-[var(--border)] bg-[var(--bg-base)] text-left text-[var(--text-secondary)]">
                      <th className="px-3 py-2 font-medium">Section</th>
                      <th className="px-3 py-2 font-medium text-right">Score</th>
                      <th className="px-3 py-2 font-medium text-right">Confidence</th>
                      <th className="px-3 py-2 font-medium text-right">Time</th>
                      <th className="px-3 py-2 font-medium text-right">Memory</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sectionRows.map((row) => (
                      <tr key={row.label} className="border-b border-[var(--border)] last:border-0">
                        <td className="px-3 py-2 text-[var(--text-primary)]">{row.label}</td>
                        <td className="px-3 py-2 text-right font-mono">
                          {formatPercent(row.scorePercent)}
                          {GT_ENABLED && row.scoreSource === 'accuracy' && (
                            <span className="ml-1 text-[var(--text-muted)]">GT</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right font-mono">
                          {formatPercent(row.confidencePercent)}
                        </td>
                        <td className="px-3 py-2 text-right font-mono">
                          {row.durationMs > 0 ? formatDurationMs(row.durationMs) : '—'}
                        </td>
                        <td className="px-3 py-2 text-right font-mono">
                          {row.memoryMb > 0 ? formatMemoryMb(row.memoryMb) : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <PerformanceRadialPanel stats={stats} totalMs={totalMs} peakMem={peakMem} />
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_220px] gap-5 items-start">
          {stages.length === 0 && (
            <p className="text-xs text-[var(--text-secondary)]">
              Per-section breakdown unavailable for this run. Re-process the resume to capture detailed stats.
            </p>
          )}
          <PerformanceRadialPanel stats={stats} totalMs={totalMs} peakMem={peakMem} />
        </div>
      )}
    </SectionCard>
  );
}
