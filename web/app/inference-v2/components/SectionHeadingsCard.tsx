import SectionCard from './SectionCard';
import type { SectionPerformanceStat } from '../types';
import { isVisibleSection } from '../lib/sectionPerformance';
import { cleanDisplayText } from '../lib/textClean';

interface Props {
  headings?: string[] | null;
  stat?: SectionPerformanceStat | null;
}

export default function SectionHeadingsCard({ headings, stat }: Props) {
  const cleaned = (headings ?? []).map(cleanDisplayText).filter((h) => h.length > 0);
  if (cleaned.length === 0) return null;
  const visibleStat = isVisibleSection(stat) ? stat : undefined;

  return (
    <SectionCard title="Section Headings" stat={visibleStat}>
      <div className="flex flex-wrap gap-2 pt-1">
        {cleaned.map((heading, idx) => (
          <span
            key={idx}
            className="inline-flex items-center px-2.5 py-1 text-xs font-mono rounded-md border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-secondary)]"
          >
            {heading}
          </span>
        ))}
      </div>
    </SectionCard>
  );
}
