'use client';

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { ModelPrecision } from '../types';

interface Props {
  value: ModelPrecision;
  onChange: (value: ModelPrecision) => void;
  disabled?: boolean;
  className?: string;
}

export const PRECISION_LABELS: Record<ModelPrecision, string> = {
  fp32: 'FP32 (full)',
  int8: 'INT8 (quantized)',
};

export default function PrecisionSelect({ value, onChange, disabled, className }: Props) {
  return (
    <Select value={value} onValueChange={(v) => onChange(v as ModelPrecision)} disabled={disabled}>
      <SelectTrigger className={className ?? 'w-[168px]'} aria-label="Model precision">
        <SelectValue placeholder="Precision" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="fp32">{PRECISION_LABELS.fp32}</SelectItem>
        <SelectItem value="int8">{PRECISION_LABELS.int8}</SelectItem>
      </SelectContent>
    </Select>
  );
}
