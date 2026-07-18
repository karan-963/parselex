import { formatPercent } from '../lib/sectionPerformance';
import type { SectionPerformanceStat } from '../types';

interface Props {
  stat?: SectionPerformanceStat | null;
  /** When true, also show score (accuracy or confidence fallback). */
  showScore?: boolean;
}

export default function SectionCardStats({ stat, showScore = false }: Props) {
  if (!stat) return null;

  const items: { label: string; value: string }[] = [];

  if (stat.durationMs > 0) {
    items.push({ label: 'Time', value: `${(stat.durationMs / 1000).toFixed(2)}s` });
  }
  if (stat.memoryMb > 0) {
    items.push({ label: 'Mem', value: `${stat.memoryMb.toFixed(0)} MB` });
  }
  if (stat.confidencePercent != null) {
    items.push({ label: 'Conf', value: formatPercent(stat.confidencePercent) });
  }
  if (showScore && stat.scorePercent != null) {
    items.push({
      label: stat.scoreSource === 'accuracy' ? 'Score' : 'Score',
      value: formatPercent(stat.scorePercent),
    });
  }

  if (items.length === 0) return null;

  return (
    <div className="flex flex-wrap justify-end gap-x-3 gap-y-1 text-[10px] font-mono text-[var(--text-secondary)]">
      {items.map((item) => (
        <span key={item.label} className="whitespace-nowrap">
          <span className="text-[var(--text-muted)]">{item.label}</span>{' '}
          <span className="text-[var(--text-primary)]">{item.value}</span>
        </span>
      ))}
    </div>
  );
}
