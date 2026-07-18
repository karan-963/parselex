import {
  HEALTH_COLORS,
  HEALTH_LABELS,
  type HealthLevel,
} from '../lib/performanceHealth';

interface Props {
  label: string;
  valueText: string;
  level: HealthLevel | null;
}

export default function PerformanceHealthBar({ label, valueText, level }: Props) {
  const color = level ? HEALTH_COLORS[level] : 'var(--text-secondary)';

  return (
    <div
      className="flex items-center justify-between gap-3 rounded-md border border-[var(--border)] bg-[var(--bg-elevated)] px-3 py-1.5"
      title={level ? `${label}: ${HEALTH_LABELS[level]}` : label}
    >
      <div className="flex items-center gap-2 min-w-0">
        <span
          className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
          style={{ backgroundColor: color, boxShadow: `0 0 0 3px ${color}22` }}
        />
        <span className="truncate text-[var(--text-secondary)]">{label}</span>
      </div>
      <span className="whitespace-nowrap font-mono font-semibold text-[var(--text-primary)]">
        {valueText}
      </span>
    </div>
  );
}
