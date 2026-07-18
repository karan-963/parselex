export type HealthLevel = 'good' | 'warn' | 'bad';

/**
 * Base transformer checkpoint size on disk (~93 MB). Runtime footprint is a
 * multiple of this because of the PyTorch/native runtime plus ~10 cached models.
 */
export const BASE_MODEL_MB = 93;

/** Resident memory health bands, expressed as multiples of the base model size. */
const MEMORY_GOOD_MB = BASE_MODEL_MB * 7; // ~650 MB — PyTorch runtime + all cached models
const MEMORY_WARN_MB = BASE_MODEL_MB * 11; // ~1 GB — elevated but tolerable

/** Total pipeline wall-time health bands (ms). */
const TIME_GOOD_MS = 6000; // cold first-resume load sits around here
const TIME_WARN_MS = 15000;

export const HEALTH_COLORS: Record<HealthLevel, string> = {
  good: '#22c55e',
  warn: '#f59e0b',
  bad: '#ef4444',
};

export const HEALTH_LABELS: Record<HealthLevel, string> = {
  good: 'In range',
  warn: 'Above expected',
  bad: 'Too high',
};

function classify(value: number, good: number, warn: number): HealthLevel {
  if (value <= good) return 'good';
  if (value <= warn) return 'warn';
  return 'bad';
}

export function memoryHealth(memMb: number | null | undefined): HealthLevel | null {
  if (memMb == null || memMb <= 0) return null;
  return classify(memMb, MEMORY_GOOD_MB, MEMORY_WARN_MB);
}

export function timeHealth(ms: number | null | undefined): HealthLevel | null {
  if (ms == null || ms <= 0) return null;
  return classify(ms, TIME_GOOD_MS, TIME_WARN_MS);
}

/** Fill ratio (0–1) for a health bar, capped so the warn ceiling reads as full. */
export function memoryFillRatio(memMb: number | null | undefined): number {
  if (memMb == null || memMb <= 0) return 0;
  return Math.min(1, memMb / MEMORY_WARN_MB);
}

export function timeFillRatio(ms: number | null | undefined): number {
  if (ms == null || ms <= 0) return 0;
  return Math.min(1, ms / TIME_WARN_MS);
}
