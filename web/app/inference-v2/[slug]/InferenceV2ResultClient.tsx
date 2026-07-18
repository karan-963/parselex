'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Loader2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import PipelineArtifactsList from '../components/PipelineArtifactsList';
import PipelinePerformanceCard from '../components/PipelinePerformanceCard';
import PrecisionSelect from '../components/PrecisionSelect';
import SectionHeadingsCard from '../components/SectionHeadingsCard';
import StructuredJsonView from '../components/StructuredJsonView';
import StructuredResumeView from '../components/StructuredResumeView';
import type { InferenceV2Run, ModelPrecision, PipelinePerformanceStats } from '../types';

interface Props {
  slug: string;
}

const POLL_MS = 2000;

export default function InferenceV2ResultClient({ slug }: Props) {
  const [run, setRun] = useState<InferenceV2Run | null>(null);
  const [tokens, setTokens] = useState<{ tokens?: unknown[] } | unknown[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [precision, setPrecision] = useState<ModelPrecision>('fp32');
  const tokensLoadedRef = useRef(false);
  const precisionInitRef = useRef(false);

  useEffect(() => {
    tokensLoadedRef.current = false;
    precisionInitRef.current = false;
    let cancelled = false;

    const fetchTokensOnce = async () => {
      if (tokensLoadedRef.current) return;
      const tokRes = await fetch(
        `/api/inference-v2/runs/${encodeURIComponent(slug)}/artifacts/14_final_classified_tokens.json`,
      );
      if (tokRes.ok && !cancelled) {
        setTokens(await tokRes.json());
        tokensLoadedRef.current = true;
      }
    };

    const poll = async (): Promise<'running' | 'done'> => {
      const res = await fetch(`/api/inference-v2/runs/${encodeURIComponent(slug)}`);
      if (!res.ok) {
        if (!cancelled) setError('Run not found');
        return 'done';
      }
      const data = (await res.json()) as InferenceV2Run;
      if (cancelled) return 'done';

      setRun(data);

      if (!precisionInitRef.current && data.modelPrecision) {
        setPrecision(data.modelPrecision);
        precisionInitRef.current = true;
      }

      if (data.status === 'completed') {
        await fetchTokensOnce();
        return 'done';
      }
      if (data.status === 'failed') {
        return 'done';
      }
      return 'running';
    };

    let intervalId: ReturnType<typeof setInterval> | undefined;

    const tick = async () => {
      const state = await poll();
      if (state === 'done' && intervalId !== undefined) {
        clearInterval(intervalId);
        intervalId = undefined;
      }
    };

    void tick();
    intervalId = setInterval(() => void tick(), POLL_MS);

    return () => {
      cancelled = true;
      if (intervalId !== undefined) clearInterval(intervalId);
    };
  }, [slug]);

  if (error) {
    return <p className="p-6 text-red-400">{error}</p>;
  }

  if (!run) {
    return (
      <div className="flex items-center justify-center min-h-[50vh] gap-2 text-[var(--text-secondary)]">
        <Loader2 className="animate-spin h-5 w-5" />
        Loading run…
      </div>
    );
  }

  const sectionHeadings = Array.isArray(run.structured?.SECTION_HEADINGS)
    ? (run.structured.SECTION_HEADINGS as string[])
    : null;

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <div className="flex items-center gap-3">
        <Button variant="outline" size="sm" asChild>
          <Link href="/inference-v2">
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back
          </Link>
        </Button>
        <div>
          <h1 className="text-xl font-bold font-mono">{run.slug}</h1>
          <p className="text-sm text-[var(--text-secondary)]">{run.originalFilename}</p>
        </div>
        <div className="ml-auto flex items-center gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-[10px] text-[var(--text-secondary)]">Re-run precision</span>
            <PrecisionSelect value={precision} onChange={setPrecision} className="w-[150px] h-8" />
          </div>
          <Badge variant={run.status === 'completed' ? 'default' : run.status === 'failed' ? 'destructive' : 'secondary'}>
            {run.status}
          </Badge>
        </div>
      </div>

      {run.status === 'running' && (
        <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)] p-4 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)]">
          <Loader2 className="animate-spin h-4 w-4" />
          Running stage: <span className="font-mono text-[var(--text-primary)]">{run.currentStage}</span>
        </div>
      )}

      {run.status === 'failed' && (
        <div className="p-4 rounded-lg border border-red-500/30 bg-red-500/10 text-sm space-y-2">
          <p>Failed at stage: <span className="font-mono">{run.failedStage}</span></p>
          <pre className="text-xs overflow-auto whitespace-pre-wrap">{run.error}</pre>
          <a
            href={`/api/inference-v2/runs/${encodeURIComponent(slug)}/artifacts/error.txt`}
            className="text-[var(--accent)] hover:underline text-xs"
          >
            Download error.txt
          </a>
        </div>
      )}

      {run.status === 'completed' && (
        <Tabs defaultValue="view">
          <TabsList>
            <TabsTrigger value="view">Structured View</TabsTrigger>
            <TabsTrigger value="structured">Structured JSON</TabsTrigger>
            <TabsTrigger value="artifacts">Artifacts</TabsTrigger>
            <TabsTrigger value="tokens">Raw Tokens</TabsTrigger>
          </TabsList>
          <TabsContent value="view" className="mt-4 space-y-4">
            <PipelinePerformanceCard
              stats={run.performanceStats as PipelinePerformanceStats | undefined}
              createdAt={run.createdAt}
              completedAt={run.completedAt}
            />
            <SectionHeadingsCard
              headings={sectionHeadings}
              stat={run.performanceStats?.sections?.section_headings}
            />
            <StructuredResumeView
              data={run.structured ?? {}}
              tokens={tokens}
              performanceStats={run.performanceStats}
            />
          </TabsContent>
          <TabsContent value="structured" className="mt-4">
            <StructuredJsonView data={run.structured ?? {}} />
          </TabsContent>
          <TabsContent value="artifacts" className="mt-4">
            <PipelineArtifactsList slug={slug} resumeId={run.resumeId} artifacts={run.artifacts ?? []} precision={precision} />
          </TabsContent>
          <TabsContent value="tokens" className="mt-4">
            <StructuredJsonView data={tokens ?? {}} />
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
