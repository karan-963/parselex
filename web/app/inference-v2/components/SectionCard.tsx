import React from 'react';
import SectionCardStats from './SectionCardStats';
import type { SectionPerformanceStat } from '../types';

interface SectionCardProps {
  title: string;
  children: React.ReactNode;
  stat?: SectionPerformanceStat | null;
}

export default function SectionCard({ title, children, stat }: SectionCardProps) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-5 mb-4">
      <div className="flex items-start justify-between gap-4 mb-2">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] shrink-0">
          Section - {title}
        </h3>
        <SectionCardStats stat={stat} />
      </div>
      <div className="border-b border-[var(--border)] mb-4" />
      <div className="space-y-1.5">{children}</div>
    </div>
  );
}
