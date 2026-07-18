'use client';

import { useCallback, useMemo, useState } from 'react';
import { Accordion } from '@/components/ui/accordion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import ArtifactJsonItem from './ArtifactJsonItem';
import type { ModelPrecision } from '../types';

interface Props {
  slug: string;
  resumeId?: string;
  artifacts: string[];
  precision?: ModelPrecision;
}

const sortArtifacts = (a: string, b: string) => {
  const matchA = a.match(/^(\d+)/);
  const matchB = b.match(/^(\d+)/);
  if (matchA && matchB) {
    const numA = parseInt(matchA[1], 10);
    const numB = parseInt(matchB[1], 10);
    if (numA !== numB) {
      return numA - numB;
    }
  } else if (matchA) {
    return -1;
  } else if (matchB) {
    return 1;
  }
  return a.localeCompare(b);
};

export default function PipelineArtifactsList({ slug, resumeId, artifacts, precision = 'fp32' }: Props) {
  const [openItems, setOpenItems] = useState<string[]>([]);

  const jsonArtifacts = useMemo(
    () => [...artifacts].filter((n) => n.endsWith('.json')).sort(sortArtifacts),
    [artifacts],
  );
  const otherArtifacts = useMemo(
    () => [...artifacts].filter((n) => !n.endsWith('.json')).sort(sortArtifacts),
    [artifacts],
  );

  const handleOpenChange = useCallback((value: string | string[]) => {
    const next = Array.isArray(value) ? value : value ? [value] : [];
    setOpenItems(next);
  }, []);

  if (artifacts.length === 0) {
    return (
      <p className="text-sm text-[var(--text-secondary)]">No artifacts available.</p>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Pipeline Artifacts</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {jsonArtifacts.length > 0 && (
          <Accordion type="multiple" value={openItems} onValueChange={handleOpenChange}>
            {jsonArtifacts.map((name) => (
              <ArtifactJsonItem
                key={name}
                slug={slug}
                resumeId={resumeId}
                filename={name}
                isOpen={openItems.includes(name)}
                precision={precision}
              />
            ))}
          </Accordion>
        )}

        {otherArtifacts.length > 0 && (
          <ul className="space-y-2 pt-2 border-t border-[var(--border)]">
            {otherArtifacts.map((name) => (
              <li key={name}>
                <a
                  href={`/api/inference-v2/runs/${encodeURIComponent(slug)}/artifacts/${encodeURIComponent(name)}`}
                  className="text-sm text-[var(--accent)] hover:underline font-mono"
                  download
                >
                  {name}
                </a>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
