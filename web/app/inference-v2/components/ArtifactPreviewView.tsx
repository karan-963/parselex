'use client';

import { useMemo } from 'react';
import {
  getArtifactPreviewConfig,
  getExcludedTopLevelKeys,
  getValueAtPath,
} from '../lib/artifactTableConfig';
import JsonArrayTable from './JsonArrayTable';

interface Props {
  filename: string;
  data: unknown;
}

function isObjectArray(value: unknown): value is Record<string, unknown>[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((row) => row !== null && typeof row === 'object' && !Array.isArray(row))
  );
}

function discoverArrayTables(data: unknown): Array<{ path: string; title: string; rows: Record<string, unknown>[] }> {
  if (!data || typeof data !== 'object' || Array.isArray(data)) return [];
  const tables: Array<{ path: string; title: string; rows: Record<string, unknown>[] }> = [];
  for (const [key, value] of Object.entries(data as Record<string, unknown>)) {
    if (isObjectArray(value)) {
      tables.push({ path: key, title: key, rows: value });
    }
  }
  return tables;
}

function JsonMetadataBlock({ value, label }: { value: unknown; label: string }) {
  return (
    <div className="space-y-1">
      <h4 className="text-xs font-semibold text-[var(--text-primary)]">{label}</h4>
      <pre className="text-[10px] overflow-auto max-h-[30vh] p-3 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border)] font-mono text-[var(--text-primary)] whitespace-pre-wrap break-words">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}

export default function ArtifactPreviewView({ filename, data }: Props) {
  const config = getArtifactPreviewConfig(filename);

  const tables = useMemo(() => {
    if (!data || typeof data !== 'object') return [];

    if (config) {
      return config.tables
        .map((spec) => {
          const value = getValueAtPath(data, spec.path);
          if (!isObjectArray(value)) return null;
          return {
            ...spec,
            rows: value,
          };
        })
        .filter(Boolean) as Array<{
        path: string;
        title: string;
        preferredColumns?: string[];
        rows: Record<string, unknown>[];
      }>;
    }

    return discoverArrayTables(data);
  }, [config, data]);

  const metadata = useMemo(() => {
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
      return data ?? {};
    }
    if (!config) {
      const excluded = new Set(tables.map((table) => table.path.split('.')[0]));
      const out: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(data as Record<string, unknown>)) {
        if (!excluded.has(key)) out[key] = value;
      }
      return out;
    }
    const excluded = getExcludedTopLevelKeys(config);
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(data as Record<string, unknown>)) {
      if (excluded.has(key)) continue;
      out[key] = value;
    }
    return out;
  }, [config, data, tables]);

  const hasMetadata =
    metadata &&
    typeof metadata === 'object' &&
    !Array.isArray(metadata) &&
    Object.keys(metadata as object).length > 0;

  if (tables.length === 0) {
    return <JsonMetadataBlock value={data} label={filename.replace('.json', '')} />;
  }

  return (
    <div className="space-y-4 my-2">
      <p className="text-[10px] text-[var(--text-secondary)]">
        Inference preview (no MongoDB ground truth). Arrays are shown as tables; other fields below.
      </p>

      {tables.map((table) => (
        <JsonArrayTable
          key={table.path}
          title={table.title}
          rows={table.rows}
          preferredColumns={table.preferredColumns}
        />
      ))}

      {hasMetadata && <JsonMetadataBlock value={metadata} label="Metadata" />}
    </div>
  );
}
