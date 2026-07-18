'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Loader2, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import RunHistoryTable from './components/RunHistoryTable';
import PrecisionSelect from './components/PrecisionSelect';
import type { InferenceV2Run, ModelPrecision } from './types';

export default function InferenceV2UploadClient() {
  const router = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);
  const [runs, setRuns] = useState<InferenceV2Run[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [precision, setPrecision] = useState<ModelPrecision>('fp32');

  const loadRuns = useCallback(async () => {
    const res = await fetch('/api/inference-v2/runs');
    if (!res.ok) return;
    const data = await res.json();
    setRuns(data.runs ?? []);
  }, []);

  useEffect(() => {
    loadRuns();
    const id = setInterval(loadRuns, 5000);
    return () => clearInterval(id);
  }, [loadRuns]);

  const startRun = async (formData?: FormData) => {
    setLoading(true);
    setError(null);
    try {
      const base = formData ? '/api/inference-v2/runs' : '/api/inference-v2/run/default';
      const url = `${base}?precision=${precision}`;
      const res = await fetch(url, formData ? { method: 'POST', body: formData } : { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? 'Failed to start inference');
      router.push(`/inference-v2/${encodeURIComponent(data.slug)}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const onUpload = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) {
      setError('Select a PDF file first');
      return;
    }
    const formData = new FormData();
    formData.append('file', file);
    await startRun(formData);
  };

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Inference V2</h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1">
          Upload a resume PDF to run the full extraction and 13-model pipeline.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Upload Resume</CardTitle>
          <CardDescription>PDF only — results saved per run for debugging.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,application/pdf"
            className="block w-full text-sm text-[var(--text-secondary)] file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:bg-[var(--bg-elevated)] file:text-[var(--text-primary)]"
          />
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex flex-col gap-1">
              <span className="text-xs text-[var(--text-secondary)]">Model precision</span>
              <PrecisionSelect value={precision} onChange={setPrecision} disabled={loading} />
            </div>
            <Button className="self-end" onClick={onUpload} disabled={loading}>
              {loading ? <Loader2 className="animate-spin mr-2 h-4 w-4" /> : <Upload className="mr-2 h-4 w-4" />}
              Run Inference
            </Button>
            <Button className="self-end" variant="outline" onClick={() => startRun()} disabled={loading}>
              Run Default (Karan.pdf)
            </Button>
          </div>
          {error && <p className="text-sm text-red-400">{error}</p>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">History</CardTitle>
        </CardHeader>
        <CardContent>
          <RunHistoryTable runs={runs} />
        </CardContent>
      </Card>
    </div>
  );
}
