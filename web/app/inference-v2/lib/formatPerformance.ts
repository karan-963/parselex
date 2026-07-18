export type {
  StagePerformanceStat,
  SectionPerformanceStat,
  PipelinePerformanceStats,
} from '../types';

export function formatDurationMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatMemoryMb(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(1)} MB`;
}

/** Fallback total duration from manifest timestamps when per-stage stats are unavailable. */
export function wallClockDurationMs(createdAt?: string, completedAt?: string | null): number | null {
  if (!createdAt || !completedAt) return null;
  const start = Date.parse(createdAt);
  const end = Date.parse(completedAt);
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) return null;
  return end - start;
}
