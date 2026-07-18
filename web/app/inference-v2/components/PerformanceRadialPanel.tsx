import PerformanceHealthBar from "./PerformanceHealthBar";
import {
  formatDurationMs,
  formatMemoryMb,
  type PipelinePerformanceStats,
} from "../lib/formatPerformance";
import { memoryHealth, timeHealth } from "../lib/performanceHealth";

interface Props {
  stats?: PipelinePerformanceStats | null;
  totalMs?: number | null;
  peakMem?: number | null;
}

export default function PerformanceRadialPanel({
  stats,
  totalMs,
  peakMem,
}: Props) {
  const conf = stats?.overallConfidencePercent;
  const confPct =
    conf == null || Number.isNaN(conf) ? null : Math.max(0, Math.min(100, conf));

  return (
    <div className="flex flex-col justify-center gap-3 rounded-lg border border-[var(--border)] bg-[var(--bg-base)] px-3 py-3 min-w-[180px]">
      <div>
        <div className="mb-1.5 flex items-baseline justify-between gap-2 text-[11px]">
          <span className="uppercase tracking-wide text-[var(--text-secondary)]">
            Confidence
          </span>
          <span className="font-mono text-base font-bold text-white">
            {confPct == null ? "—" : `${confPct.toFixed(0)}%`}
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--bg-elevated)]">
          <div
            className="h-full rounded-full bg-[var(--accent)] transition-[width] duration-500"
            style={{ width: `${confPct ?? 0}%` }}
          />
        </div>
      </div>
      <div className="w-full space-y-1.5 border-t border-[var(--border)] pt-3 text-[11px]">
        {totalMs != null && (
          <PerformanceHealthBar
            label="Time"
            valueText={formatDurationMs(totalMs)}
            level={timeHealth(totalMs)}
          />
        )}
        {peakMem != null && peakMem > 0 && (
          <PerformanceHealthBar
            label="Memory"
            valueText={formatMemoryMb(peakMem)}
            level={memoryHealth(peakMem)}
          />
        )}
      </div>
    </div>
  );
}
