'use client';

import { useMemo, useState } from 'react';
import { Search } from 'lucide-react';

interface Props {
  title: string;
  rows: Record<string, unknown>[];
  preferredColumns?: string[];
  maxHeight?: string;
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function deriveColumns(rows: Record<string, unknown>[], preferred?: string[]): string[] {
  if (!rows.length) return preferred ?? [];
  const keys = new Set<string>();
  for (const row of rows.slice(0, 50)) {
    Object.keys(row).forEach((key) => keys.add(key));
  }
  const discovered = [...keys];
  if (!preferred?.length) return discovered;
  const ordered = preferred.filter((col) => keys.has(col));
  for (const col of discovered) {
    if (!ordered.includes(col)) ordered.push(col);
  }
  return ordered;
}

export default function JsonArrayTable({
  title,
  rows,
  preferredColumns,
  maxHeight = '50vh',
}: Props) {
  const [search, setSearch] = useState('');
  const columns = useMemo(() => deriveColumns(rows, preferredColumns), [rows, preferredColumns]);

  const filteredRows = useMemo(() => {
    if (!search.trim()) return rows;
    const q = search.toLowerCase();
    return rows.filter((row) =>
      columns.some((col) => formatCell(row[col]).toLowerCase().includes(q)),
    );
  }, [rows, columns, search]);

  if (!rows.length) {
    return (
      <div className="text-xs text-[var(--text-secondary)] border border-[var(--border)] rounded-lg p-3">
        {title}: no rows
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-xs font-semibold text-[var(--text-primary)]">
          {title}
          <span className="ml-2 font-normal text-[var(--text-secondary)]">
            ({filteredRows.length}/{rows.length})
          </span>
        </h4>
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-[var(--text-secondary)]" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter rows…"
            className="pl-7 pr-2 py-1 text-[10px] rounded border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-primary)] w-40"
          />
        </div>
      </div>
      <div
        className="overflow-auto border border-[var(--border)] rounded-lg"
        style={{ maxHeight }}
      >
        <table className="w-full text-left border-collapse text-xs">
          <thead className="bg-[var(--bg-elevated)] sticky top-0 border-b border-[var(--border)] font-mono text-[10px] uppercase tracking-wider text-[var(--text-secondary)]">
            <tr>
              <th className="p-2 font-semibold w-10">#</th>
              {columns.map((col) => (
                <th key={col} className="p-2 font-semibold whitespace-nowrap">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono text-[11px] divide-y divide-[var(--border)]/50">
            {filteredRows.map((row, index) => (
              <tr key={index} className="hover:bg-[var(--bg-elevated)]/50">
                <td className="p-2 text-[var(--text-secondary)]">{index + 1}</td>
                {columns.map((col) => (
                  <td
                    key={col}
                    className="p-2 text-[var(--text-primary)] max-w-xs truncate"
                    title={formatCell(row[col])}
                  >
                    {formatCell(row[col])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
