'use client';

import Link from 'next/link';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { InferenceV2Run } from '../types';

function statusVariant(status: InferenceV2Run['status']) {
  if (status === 'completed') return 'default';
  if (status === 'failed') return 'destructive';
  return 'secondary';
}

function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

interface Props {
  runs: InferenceV2Run[];
}

export default function RunHistoryTable({ runs }: Props) {
  if (runs.length === 0) {
    return (
      <p className="text-sm text-[var(--text-secondary)] py-8 text-center">
        No inference runs yet. Upload a PDF or run the default Karan test.
      </p>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Run</TableHead>
          <TableHead>File</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Created</TableHead>
          <TableHead className="text-right">Action</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {runs.map((run) => (
          <TableRow key={run.slug}>
            <TableCell className="font-mono text-xs">{run.slug}</TableCell>
            <TableCell className="max-w-[200px] truncate">{run.originalFilename}</TableCell>
            <TableCell>
              <Badge variant={statusVariant(run.status)}>{run.status}</Badge>
            </TableCell>
            <TableCell className="text-xs text-[var(--text-secondary)]">
              {formatDate(run.createdAt)}
            </TableCell>
            <TableCell className="text-right">
              <Link
                href={`/inference-v2/${encodeURIComponent(run.slug)}`}
                className="text-sm text-[var(--accent)] hover:underline"
              >
                View
              </Link>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
