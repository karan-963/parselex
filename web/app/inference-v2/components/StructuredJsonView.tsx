'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface Props {
  data: unknown;
}

export default function StructuredJsonView({ data }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Structured JSON</CardTitle>
      </CardHeader>
      <CardContent>
        <pre className="text-xs overflow-auto max-h-[70vh] p-4 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border)]">
          {JSON.stringify(data, null, 2)}
        </pre>
      </CardContent>
    </Card>
  );
}
