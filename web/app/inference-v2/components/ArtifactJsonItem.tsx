'use client';

import { useState } from 'react';
import { Loader2, RotateCw } from 'lucide-react';
import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import ArtifactJsonLoader from './ArtifactJsonLoader';
import { formatArtifactAccordionHint, formatArtifactAccordionLabel } from '../lib/artifactCatalog';
import { getRerunStage } from '../lib/rerunStages';
import type { ModelPrecision } from '../types';

interface Props {
  slug: string;
  resumeId?: string;
  filename: string;
  isOpen: boolean;
  precision?: ModelPrecision;
}

export default function ArtifactJsonItem({ slug, resumeId, filename, isOpen, precision = 'fp32' }: Props) {
  const downloadUrl = `/api/inference-v2/runs/${encodeURIComponent(slug)}/artifacts/${encodeURIComponent(filename)}`;

  const label = formatArtifactAccordionLabel(filename);
  const hint = formatArtifactAccordionHint(filename);

  const rerunStage = getRerunStage(filename);
  const [rerunning, setRerunning] = useState(false);
  const [rerunError, setRerunError] = useState<string | null>(null);
  const [reloadTick, setReloadTick] = useState(0);

  const handleRerun = async () => {
    if (!rerunStage || rerunning) return;
    setRerunning(true);
    setRerunError(null);
    try {
      const res = await fetch(
        `/api/inference-v2/runs/${encodeURIComponent(slug)}/rerun/${encodeURIComponent(rerunStage)}?precision=${precision}`,
        { method: 'POST' },
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data?.detail || data?.error || `Rerun failed (${res.status})`);
      }
      setReloadTick((n) => n + 1);
    } catch (err) {
      setRerunError(err instanceof Error ? err.message : String(err));
    } finally {
      setRerunning(false);
    }
  };

  return (
    <AccordionItem value={filename} className="border border-[var(--border)] rounded-lg px-4 mb-2 last:border-b">
      <AccordionTrigger className="font-mono text-xs hover:no-underline py-3">
        <span className="truncate text-left flex flex-col items-start gap-0.5">
          <span className="font-sans font-medium text-[var(--text-primary)]">{label}</span>
          <span className="text-[10px] text-[var(--text-secondary)] font-mono">{filename}</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        <div className="space-y-2 pb-2">
          {hint && (
            <p className="text-[10px] text-[var(--text-secondary)]">{hint}</p>
          )}
          <div className="flex items-center gap-3">
            <a
              href={downloadUrl}
              download
              className="text-xs text-[var(--accent)] hover:underline"
            >
              Download
            </a>
            {rerunStage && (
              <button
                type="button"
                onClick={handleRerun}
                disabled={rerunning}
                className="inline-flex items-center gap-1 px-2 py-1 rounded border border-[var(--border)] bg-[var(--bg-elevated)] text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-50"
                title={`Re-run this step (${rerunStage}) and reload its artifact`}
              >
                {rerunning ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <RotateCw className="h-3 w-3" />
                )}
                {rerunning ? 'Re-running…' : 'Re-run step'}
              </button>
            )}
          </div>
          {rerunError && <p className="text-[10px] text-red-400">{rerunError}</p>}
          {isOpen ? (
            <ArtifactJsonLoader key={reloadTick} slug={slug} resumeId={resumeId} filename={filename} />
          ) : null}
        </div>
      </AccordionContent>
    </AccordionItem>
  );
}
