import type { SectionPerformanceStat } from '../types';
import { formatDurationMs, formatMemoryMb } from './formatPerformance';

export function formatPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '—';
  return `${value.toFixed(1)}%`;
}

export function sectionStatKeyForResumeType(
  type: 'profile' | 'experience' | 'education' | 'projects' | 'skills' | 'headings',
): keyof Record<string, SectionPerformanceStat> | null {
  const map = {
    profile: 'profile',
    experience: 'experience',
    education: 'education',
    projects: 'projects',
    skills: 'skills',
    headings: 'section_headings',
  } as const;
  return map[type];
}

export function isVisibleSection(row?: SectionPerformanceStat | null): boolean {
  if (!row) return false;
  if (row.present === false) return false;
  if (row.present === true) return true;
  return row.scorePercent != null || row.confidencePercent != null;
}

export function statBadgeItems(stat?: SectionPerformanceStat | null) {
  if (!stat) return null;
  return {
    time: stat.durationMs > 0 ? formatDurationMs(stat.durationMs) : null,
    memory: stat.memoryMb > 0 ? formatMemoryMb(stat.memoryMb) : null,
    confidence: formatPercent(stat.confidencePercent),
    score: formatPercent(stat.scorePercent),
  };
}
