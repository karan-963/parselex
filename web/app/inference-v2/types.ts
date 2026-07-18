export type ModelPrecision = 'fp32' | 'int8';

export interface StagePerformanceStat {
  stage: string;
  label: string;
  durationMs: number;
  memoryMb: number;
  memoryDeltaMb: number;
}

export interface SectionPerformanceStat {
  label: string;
  present?: boolean;
  durationMs: number;
  memoryMb: number;
  confidencePercent: number | null;
  scorePercent: number | null;
  scoreSource: 'accuracy' | 'confidence' | null;
  stages: string[];
}

export interface PipelinePerformanceStats {
  totalDurationMs: number;
  peakMemoryMb: number;
  overallScorePercent?: number | null;
  overallConfidencePercent?: number | null;
  stages: StagePerformanceStat[];
  sections?: Record<string, SectionPerformanceStat>;
}

export interface InferenceV2Run {
  slug: string;
  originalFilename: string;
  modelPrecision?: ModelPrecision;
  status: 'running' | 'completed' | 'failed';
  currentStage?: string;
  failedStage?: string | null;
  error?: string | null;
  createdAt: string;
  updatedAt?: string;
  completedAt?: string | null;
  artifacts?: string[];
  structured?: Record<string, unknown>;
  resumeId?: string;
  performanceStats?: PipelinePerformanceStats;
}
