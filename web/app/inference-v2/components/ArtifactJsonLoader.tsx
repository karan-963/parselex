'use client';

import { useEffect, useState, useMemo } from 'react';
import { Loader2, RefreshCw, Search, SlidersHorizontal } from 'lucide-react';
import EntrySectionDividerView from './EntrySectionDividerView';
import FieldClassificationView from './FieldClassificationView';
import ArtifactPreviewView from './ArtifactPreviewView';
import { applyStep8GroundTruthLabels, STEP8_GT_SOURCE } from '../lib/step8BoundaryGt';
import { isGtEnabled } from '../lib/gtGate';
import { applyStep12GroundTruthLabels } from '../lib/step12BoundaryGt';
import {
  computeSegmentEvalStats,
  isEvalSegToken,
  mapSkillsLabelTo5Class,
  resolveSegmentGtToken,
} from '../lib/segmentTokenAlignment';

interface Props {
  slug: string;
  resumeId?: string;
  filename: string;
}

interface AlignedRow {
  page: number;
  lineIndex: number;
  tokenIndex: number;
  token: string | null;
  infLabel: string;
  gtLabel: string;
  gtToken: string | null;
  mongoBioLabel?: string | null;
  status: string;
  coords: string;
  section: string;
}


const roundCoord = (n: number) => Math.round(n * 100) / 100;

const coordKey = (t: any) =>
  `${t.page}-${roundCoord(t.x0 ?? 0)}-${roundCoord(t.y0 ?? 0)}`;

const getGtSegmentLabel = (t: any) => {
  const tok = (t.token ?? '').trim();
  const structural = new Set(['|', '•', '-', '–', '—', '*', '▪', '◦', '■', '·', ',', '✓', '✔']);
  if (structural.has(tok) || tok === '"') return 'O';
  if (!t.bioLabel || t.bioLabel === 'O' || t.bioLabel.includes('HEADING')) return 'O';
  return t.bioLabel.startsWith('B-') ? 'B-SEG' : 'I-SEG';
};

const hasAlphanumeric = (text: string | null): boolean =>
  !!text && /[a-zA-Z0-9]/.test(text);

/** Strip leading/trailing punctuation for split-vs-merged token matching. */
const alnumCore = (text: string | null): string => {
  if (!text) return '';
  return text.replace(/^[^a-zA-Z0-9]+/, '').replace(/[^a-zA-Z0-9]+$/, '');
};

/** True when inference token is a PDF split of a merged MongoDB token on the same line. */
const isSplitTokenAlignment = (infToken: string | null, gtToken: string | null): boolean => {
  if (!infToken || !gtToken || infToken === gtToken) return false;
  if (!hasAlphanumeric(infToken) && gtToken.startsWith(infToken.trim())) return true;
  return alnumCore(infToken) === alnumCore(gtToken);
};

/** Resolve GT token by coordinate or merged-token text on the same line. */
const resolveGtToken = (
  inf: any,
  dbByCoord: Map<string, any>,
  sectionDbTokens: any[],
  matchedGtCoords: Set<string>,
): any | null => {
  const key = coordKey(inf);
  const primary = dbByCoord.get(key) ?? null;
  if (primary && !matchedGtCoords.has(key)) return primary;

  const lineIdx = inf.lineIndex ?? inf.line_index ?? 0;

  // Punctuation prefix of merged DB token, e.g. "(" vs "(March" (same x0,y0 or prefix)
  if (inf.token && !hasAlphanumeric(inf.token)) {
    const prefix = inf.token.trim();
    const byPrefix = sectionDbTokens.find(
      (t) =>
        t.page === inf.page &&
        (t.lineIndex ?? t.line_index ?? 0) === lineIdx &&
        (t.token ?? '').startsWith(prefix) &&
        t.token !== inf.token,
    );
    if (byPrefix) return byPrefix;
  }

  if (inf.token && hasAlphanumeric(inf.token)) {
    const core = alnumCore(inf.token);
    return (
      sectionDbTokens.find(
        (t) =>
          t.page === inf.page &&
          (t.lineIndex ?? t.line_index ?? 0) === lineIdx &&
          alnumCore(t.token) === core &&
          t.token !== inf.token,
      ) ?? null
    );
  }

  return null;
};

export default function ArtifactJsonLoader({ slug, resumeId, filename }: Props) {
  const [data, setData] = useState<any>(null);
  const [dbResumeData, setDbResumeData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter and search state
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'ERRORS' | 'MATCHES'>('ALL');
  const [gtRefreshTick, setGtRefreshTick] = useState(0);
  const [gtRefreshing, setGtRefreshing] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    const url = `/api/inference-v2/runs/${encodeURIComponent(slug)}/artifacts/${encodeURIComponent(filename)}`;

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load (${res.status})`);
        return res.json();
      })
      .then((json) => {
        if (!cancelled) setData(json);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [slug, filename]);

  useEffect(() => {
    if (!resumeId || !isGtEnabled(resumeId)) {
      setDbResumeData(null);
      return;
    }

    let cancelled = false;
    setGtRefreshing(true);

    fetch(`/api/resumes/${encodeURIComponent(resumeId)}`, { cache: 'no-store' })
      .then((dbRes) => (dbRes.ok ? dbRes.json() : null))
      .then((dbData) => {
        if (!cancelled && dbData) setDbResumeData(dbData);
      })
      .catch((dbErr) => {
        console.warn('Failed to load ground truth resume:', dbErr);
      })
      .finally(() => {
        if (!cancelled) setGtRefreshing(false);
      });

    return () => {
      cancelled = true;
    };
  }, [resumeId, gtRefreshTick]);

  // Extract inference tokens
  const infTokens = useMemo(() => {
    if (!data) return [];
    if (Array.isArray(data.tokens)) return data.tokens;
    if (Array.isArray(data)) return data;
    return [];
  }, [data]);

  // Build aligned comparison rows
  const comparisonRows = useMemo(() => {
    if (!dbResumeData || !Array.isArray(dbResumeData.tokens)) {
      return [];
    }

    const dbTokens = dbResumeData.tokens;
    const nameLower = filename.toLowerCase();

    if (nameLower !== '2_section_headings.json' && nameLower !== '3_section_labels.json' && infTokens.length === 0) {
      return [];
    }

    // ── 3_section_labels.json ──────────────────────────────────────────────────
    if (nameLower === '3_section_labels.json') {
      const sourceTokens = dbResumeData.pipelineTokens || dbTokens;


      // Build GT section map: heading y0 → { section, tokenCount }
      const headingTokens = sourceTokens.filter((t: any) => t.bioLabel === 'B-HEADING' || t.bioLabel === 'I-HEADING');
      const pipelineGroups = new Map<string, any[]>();
      for (const t of headingTokens) {
        const key = `${t.page}-${t.lineIndex}`;
        if (!pipelineGroups.has(key)) pipelineGroups.set(key, []);
        pipelineGroups.get(key)!.push(t);
      }

      type GtSection = { page: number; lineIndex: number; headingText: string; section: string; y0: number; tokenCount: number };
      const gtSections: GtSection[] = [];
      for (const [, group] of pipelineGroups.entries()) {
        group.sort((a: any, b: any) => a.tokenIndex - b.tokenIndex);
        const page = group[0].page;
        const lineIndex = group[0].lineIndex;
        const headingText = group.map((t: any) => t.token).join(' ');
        const y0 = group[0].y0;
        const section = group[0].section ?? '?';
        // Count all tokens in pipeline with same section label (excluding PERSONAL)
        const tokenCount = sourceTokens.filter((t: any) => t.section === section && t.bioLabel !== 'B-HEADING' && t.bioLabel !== 'I-HEADING').length;
        gtSections.push({ page, lineIndex, headingText, section, y0, tokenCount });
      }

      // Also get PERSONAL section token count (no heading in pipeline)
      const personalCount = sourceTokens.filter((t: any) => t.section === 'PERSONAL').length;

      const Y0_TOL = 3;
      const infChunks: any[] = data.chunks || [];
      const matchedGtKeys = new Set<string>();
      const rows: AlignedRow[] = [];

      for (const chunk of infChunks) {
        // Find the matching GT heading by scanning headings json if available, or fall back to text match
        const headingsData = dbResumeData.pipelineTokens
          ? null // we'll match via headingText
          : null;
        void headingsData;

        // Match by heading text (case-insensitive) since chunks don't have y0
        const gt = gtSections.find(
          (g) => g.headingText.toLowerCase().trim() === (chunk.heading || '').toLowerCase().trim()
        );
        const gtKey = gt ? `${gt.page}-${gt.lineIndex}` : null;
        if (gtKey) matchedGtKeys.add(gtKey);

        const sectionMatch = gt ? chunk.section === gt.section : false;
        const status = !gt
          ? '❌ MISMATCH'
          : sectionMatch
            ? '✅ MATCH'
            : `❌ SECTION MISMATCH (GT: ${gt.section})`;

        rows.push({
          page: gt?.page ?? 0,
          lineIndex: gt?.lineIndex ?? 0,
          tokenIndex: 0,
          token: chunk.heading || '(no heading)',
          infLabel: chunk.section,
          gtLabel: gt?.section ?? '—',
          gtToken: gt?.headingText ?? null,
          status,
          coords: `Conf: ${(chunk.confidence * 100).toFixed(1)}% | Tokens: ${chunk.tokenCount}${gt ? ` / GT: ${gt.tokenCount}` : ''}`,
          section: chunk.section,
        });
      }

      // GT sections not matched by any chunk
      for (const gt of gtSections) {
        const gtKey = `${gt.page}-${gt.lineIndex}`;
        if (!matchedGtKeys.has(gtKey)) {
          rows.push({
            page: gt.page,
            lineIndex: gt.lineIndex,
            tokenIndex: 0,
            token: null,
            infLabel: '—',
            gtLabel: gt.section,
            gtToken: gt.headingText,
            status: '❌ NOT EXTRACTED',
            coords: `GT Tokens: ${gt.tokenCount}`,
            section: gt.section,
          });
        }
      }

      // Add PERSONAL as a synthetic row (always first, no heading)
      rows.unshift({
        page: 0,
        lineIndex: 0,
        tokenIndex: 0,
        token: '(document header)',
        infLabel: 'PERSONAL',
        gtLabel: 'PERSONAL',
        gtToken: '(document header)',
        status: personalCount > 0 ? '✅ MATCH' : '❌ MISMATCH',
        coords: `Tokens: ${data.sectionTokenCounts?.PERSONAL ?? '?'} / GT: ${personalCount}`,
        section: 'PERSONAL',
      });

      rows.sort((a, b) => {
        if (a.section === 'PERSONAL') return -1;
        if (b.section === 'PERSONAL') return 1;
        if (a.page !== b.page) return a.page - b.page;
        return a.lineIndex - b.lineIndex;
      });

      return rows;
    }

    if (nameLower === '2_section_headings.json') {
      // Build GT headings keyed by page + y0 coordinate (since inference uses local lineIndex
      // and pipeline uses global lineIndex — they differ, but y0 is stable across both).
      const sourceTokens = dbResumeData.pipelineTokens || dbTokens;
      const headingsTokens = sourceTokens.filter((t: any) => t.bioLabel === 'B-HEADING' || t.bioLabel === 'I-HEADING');
      
      // Group pipeline heading tokens by page + lineIndex
      const pipelineGroups = new Map<string, any[]>();
      for (const t of headingsTokens) {
        const key = `${t.page}-${t.lineIndex}`;
        if (!pipelineGroups.has(key)) pipelineGroups.set(key, []);
        pipelineGroups.get(key)!.push(t);
      }
      
      // Build a list of GT headings with y0 for coordinate-based matching
      type GtHeading = { page: number; lineIndex: number; text: string; y0: number };
      const gtHeadings: GtHeading[] = [];
      for (const [, group] of pipelineGroups.entries()) {
        group.sort((a, b) => a.tokenIndex - b.tokenIndex);
        const page = group[0].page;
        const lineIndex = group[0].lineIndex;
        const text = group.map((t) => t.token).join(' ');
        const y0 = group[0].y0;
        gtHeadings.push({ page, lineIndex, text, y0 });
      }
      
      // Match inference heading to GT by page + y0 within 3pt tolerance
      const Y0_TOL = 3;
      const findGt = (infPage: number, infY0: number): GtHeading | undefined => {
        return gtHeadings.find(
          (g) => g.page === infPage && Math.abs(g.y0 - infY0) <= Y0_TOL
        );
      };
      
      const rows: AlignedRow[] = [];
      const infHeadings = data.headings || [];
      const matchedGtKeys = new Set<string>();
      
      for (const inf of infHeadings) {
        const gt = findGt(inf.page, inf.y0 ?? 0);
        const gtKey = gt ? `${gt.page}-${gt.lineIndex}` : null;
        if (gtKey) matchedGtKeys.add(gtKey);
        
        let gtLabel = '—';
        let status = '❌ MISMATCH';
        let gtToken = null;
        
        if (gt) {
          gtLabel = gt.text;
          gtToken = gt.text;
          status = inf.text.toLowerCase().trim() === gt.text.toLowerCase().trim() ? '✅ MATCH' : '⚠️ WORD MISMATCH';
        }
        
        rows.push({
          page: inf.page,
          lineIndex: inf.lineIndex,
          tokenIndex: 0,
          token: inf.text,
          infLabel: `${inf.source}${inf.model_prob ? ` (${(inf.model_prob * 100).toFixed(0)}%)` : ''}`,
          gtLabel: gtLabel,
          gtToken: gtToken,
          status,
          coords: inf.x0 != null ? `(${inf.x0.toFixed(1)}, ${inf.y0?.toFixed(1) ?? 0})` : `(L${inf.lineIndex})`,
          section: 'HEADING',
        });
      }
      
      // Any GT headings not matched → NOT EXTRACTED
      for (const gt of gtHeadings) {
        const gtKey = `${gt.page}-${gt.lineIndex}`;
        if (!matchedGtKeys.has(gtKey)) {
          rows.push({
            page: gt.page,
            lineIndex: gt.lineIndex,
            tokenIndex: 0,
            token: null,
            infLabel: '—',
            gtLabel: gt.text,
            gtToken: gt.text,
            status: '❌ NOT EXTRACTED',
            coords: `(y0=${gt.y0.toFixed(1)})`,
            section: 'HEADING',
          });
        }
      }
      
      rows.sort((a, b) => {
        if (a.page !== b.page) return a.page - b.page;
        return a.lineIndex - b.lineIndex;
      });
      
      return rows;
    }
    let section = '';
    if (nameLower.includes('education')) section = 'EDUCATION';
    else if (nameLower.includes('experience')) section = 'EXPERIENCE';
    else if (nameLower.includes('project')) section = 'PROJECT';
    else if (nameLower.includes('personal')) section = 'PERSONAL';
    else if (nameLower.includes('skills')) section = 'SKILLS';

    // Always use MongoDB dbTokens as the ground truth source so edits immediately reflect in visualizer
    const gtSourceTokens = dbTokens;

    // 1. Filter GT tokens by section if applicable
    let sectionDbTokens = section
      ? gtSourceTokens.filter((t: any) =>
          section === 'PROJECT'
            ? t.section === 'PROJECT' || t.section === 'PROJECTS'
            : t.section === section,
        )
      : gtSourceTokens;

    const isSegOrBoundaryOrEntity = nameLower.includes('segments') || nameLower.includes('boundaries') || nameLower.includes('entities') || nameLower.includes('divider') || nameLower.includes('fields');
    if (isSegOrBoundaryOrEntity) {
      sectionDbTokens = sectionDbTokens.filter(
        (t: any) => t.bioLabel !== 'B-HEADING' && t.bioLabel !== 'I-HEADING'
      );
    }

    // Step 9 (experience) & 12 (project) boundary GT mapping (MongoDB bioLabel only)
    if (nameLower.includes('experience_boundaries')) {
      applyStep8GroundTruthLabels(sectionDbTokens, '_temp_gtLabel');
    } else if (nameLower.includes('project_boundaries')) {
      applyStep12GroundTruthLabels(sectionDbTokens, '_temp_gtLabel');
    } else for (const gt of sectionDbTokens) {
      if (nameLower.includes('segments')) {
        gt._temp_gtLabel = getGtSegmentLabel(gt);
      } else if (nameLower === '7_skills_fields.json') {
        gt._temp_gtLabel = mapSkillsLabelTo5Class(gt.bioLabel);
      } else {
        gt._temp_gtLabel = gt.bioLabel || 'O';
      }
    }

    // Pre-calculate raw inference labels
    for (const inf of infTokens) {
      if (nameLower.includes('boundaries') || nameLower.includes('segments') || nameLower.includes('entities') || nameLower.includes('divider') || nameLower.includes('fields')) {
        inf._temp_infLabel = inf.prediction || 'O';
      } else {
        inf._temp_infLabel = inf.bioLabel || inf.bio_label || 'O';
      }
    }

    // Coordinate-key map for alignment (page, round(x0,2), round(y0,2))
    const dbByCoord = new Map<string, any>();
    for (const t of sectionDbTokens) {
      dbByCoord.set(coordKey(t), t);
    }

    const isSegmentArtifact = nameLower.includes('segments');
    const isSkillsFieldsArtifact = nameLower === '7_skills_fields.json';
    const matchedGtCoords = new Set<string>();

    const rows: AlignedRow[] = [];

    // Align inference tokens to MongoDB by coordinate key
    for (const inf of infTokens) {
      const key = coordKey(inf);
      const gt = isSegmentArtifact
        ? resolveSegmentGtToken(inf, dbByCoord, sectionDbTokens)
        : resolveGtToken(inf, dbByCoord, sectionDbTokens, matchedGtCoords);
      if (gt) {
        matchedGtCoords.add(coordKey(gt));
      }

      let gtLabel = '—';
      let infLabel = inf._temp_infLabel || 'O';

      if (gt) {
        gtLabel = gt._temp_gtLabel || 'O';
      }

      let status = '❌ NOT IN DB';
      if (gt) {
        if (nameLower === '1_extracted_tokens.json') {
          status =
            inf.token === gt.token
              ? '✅ MATCH'
              : isSplitTokenAlignment(inf.token, gt.token)
                ? '✅ MATCH (split)'
                : '⚠️ WORD MISMATCH';
        } else if ((isSegmentArtifact || isSkillsFieldsArtifact) && !isEvalSegToken(inf.token)) {
          status = '✅ MATCH (non-eval)';
        } else {
          const labelsMatch = infLabel === gtLabel;
          status = labelsMatch ? '✅ MATCH' : '❌ MISMATCH';
        }
      } else {
        const anyGt = gtSourceTokens.find((t: any) => coordKey(t) === key);
        if (anyGt) {
          status = isSegOrBoundaryOrEntity
            ? `❌ SECTION MISMATCH (GT: ${anyGt.section})`
            : `❌ SECTION MISMATCH (${anyGt.section})`;
        }
      }

      rows.push({
        page: inf.page,
        lineIndex: inf.lineIndex ?? inf.line_index ?? 0,
        tokenIndex: inf.tokenIndex ?? inf.token_index ?? 0,
        token: inf.token,
        infLabel,
        gtLabel,
        gtToken: gt ? gt.token : null,
        mongoBioLabel: gt ? (gt.bioLabel ?? null) : null,
        status,
        coords: `(${inf.x0?.toFixed(1) ?? '?'}, ${inf.y0?.toFixed(1) ?? '?'})`,
        section: gt ? gt.section : '—',
      });
    }

    // Detect DB tokens not matched by any inference token
    for (const gt of sectionDbTokens) {
      const key = coordKey(gt);
      if (!matchedGtCoords.has(key)) {
        const gtLabel = gt._temp_gtLabel || '—';

        rows.push({
          page: gt.page,
          lineIndex: gt.lineIndex ?? 0,
          tokenIndex: gt.tokenIndex ?? 0,
          token: null,
          infLabel: '—',
          gtLabel,
          gtToken: gt.token,
          status: '❌ NOT EXTRACTED',
          coords: `(${gt.x0.toFixed(1)}, ${gt.y0.toFixed(1)})`,
          section: gt.section,
        });
      }
    }

    // 6. Sort rows sequentially (page → line → token)
    rows.sort((a, b) => {
      if (a.page !== b.page) return a.page - b.page;
      if (a.lineIndex !== b.lineIndex) return a.lineIndex - b.lineIndex;
      return a.tokenIndex - b.tokenIndex;
    });

    return rows;
  }, [infTokens, dbResumeData, filename]);

  // Compute filtered rows
  const filteredRows = useMemo(() => {
    return comparisonRows.filter((row) => {
      // 1. Text Search
      const searchMatch = searchQuery
        ? (row.token || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
          (row.gtToken || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
          row.infLabel.toLowerCase().includes(searchQuery.toLowerCase()) ||
          row.gtLabel.toLowerCase().includes(searchQuery.toLowerCase())
        : true;

      if (!searchMatch) return false;

      // 2. Status Filter
      if (statusFilter === 'ERRORS') {
        return row.status !== '✅ MATCH' && !row.status.includes('non-eval') && !row.status.includes('split');
      }
      if (statusFilter === 'MATCHES') {
        return row.status === '✅ MATCH' || row.status.includes('non-eval') || row.status.includes('split');
      }
      return true;
    });
  }, [comparisonRows, searchQuery, statusFilter]);

  // Stats summaries
  const stats = useMemo(() => {
    if (comparisonRows.length === 0) return null;
    const nameLower = filename.toLowerCase();
    const isSegmentArtifact = nameLower.includes('segments');
    const headingsMode = ['2_section_headings.json', '3_section_labels.json'].includes(nameLower) ||
      nameLower.includes('segments') || nameLower.includes('boundaries') || nameLower.includes('entities') || nameLower.includes('divider') || nameLower.includes('fields');

    if (isSegmentArtifact) {
      return computeSegmentEvalStats(
        comparisonRows,
        data?.tokenSegmentation?.metrics,
      );
    }

    if (filename === '7_skills_fields.json') {
      return computeSegmentEvalStats(
        comparisonRows,
        data?.tokenClassification?.metrics,
      );
    }

    const total = comparisonRows.length;
    const matches = comparisonRows.filter((r) => r.status === '✅ MATCH').length;
    const mismatches = comparisonRows.filter((r) => r.status === '❌ MISMATCH' || r.status === '⚠️ WORD MISMATCH' || r.status.startsWith('❌ SECTION MISMATCH') || r.status.startsWith('❌ MISMATCH')).length;
    const notInDb = headingsMode ? 0 : comparisonRows.filter((r) => r.status === '❌ NOT IN DB' || r.status.startsWith('❌ SECTION MISMATCH')).length;
    const notExtracted = comparisonRows.filter((r) => r.status === '❌ NOT EXTRACTED').length;

    return {
      total,
      matches,
      mismatches,
      notInDb,
      notExtracted,
      evalTokens: total,
      headingsMode,
      accuracy: total > 0 ? ((matches / total) * 100).toFixed(1) : '0',
    };
  }, [comparisonRows, filename, data?.tokenSegmentation?.metrics, data?.tokenClassification?.metrics]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)] py-4">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading…
      </div>
    );
  }

  if (error) {
    return <p className="text-xs text-red-400">{error}</p>;
  }

  const isExperienceBoundaryArtifact =
    filename === '9_experience_boundaries.json' ||
    filename === '8_experience_boundaries.json' ||
    (filename.toLowerCase().includes('experience') && filename.toLowerCase().includes('boundaries'));

  const isProjectBoundaryArtifact =
    filename === '12_project_boundaries.json' ||
    (filename.toLowerCase().includes('project') && filename.toLowerCase().includes('boundaries'));

  const isEducationBoundaryArtifact =
    filename === '5_education_boundaries.json' ||
    (filename.toLowerCase().includes('education') && filename.toLowerCase().includes('boundaries'));

  if ((isExperienceBoundaryArtifact || isProjectBoundaryArtifact || isEducationBoundaryArtifact) && data) {
    if (resumeId) {
      return (
        <EntrySectionDividerView
          slug={slug}
          resumeId={resumeId}
          filename={filename}
          artifact={data}
        />
      );
    }
    return <ArtifactPreviewView filename={filename} data={data} />;
  }

  const isFieldClassificationArtifact =
    filename === '6_education_fields.json' ||
    filename === '10_experience_classification.json' ||
    filename === '10_experience_fields.json' ||
    filename === '13_project_fields.json' ||
    filename === '15_personal_fields.json' ||
    (filename.toLowerCase().includes('education') && filename.toLowerCase().includes('fields')) ||
    (filename.toLowerCase().includes('experience') && filename.toLowerCase().includes('fields')) ||
    (filename.toLowerCase().includes('project') && filename.toLowerCase().includes('fields')) ||
    (filename.toLowerCase().includes('personal') && filename.toLowerCase().includes('fields'));

  if (isFieldClassificationArtifact && data) {
    return <FieldClassificationView slug={slug} resumeId={resumeId} artifact={data} />;
  }

  const renderStatusBadge = (status: string) => {
    switch (status) {
      case '✅ MATCH':
        return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium font-mono bg-green-500/10 text-green-400 border border-green-500/20">{status}</span>;
      case '❌ MISMATCH':
        return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium font-mono bg-red-500/10 text-red-400 border border-red-500/20">{status}</span>;
      case '⚠️ WORD MISMATCH':
        return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium font-mono bg-yellow-500/10 text-yellow-400 border border-yellow-500/20">{status}</span>;
      case '❌ NOT IN DB':
        return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium font-mono bg-purple-500/10 text-purple-400 border border-purple-500/20">{status}</span>;
      case '❌ NOT EXTRACTED':
        return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium font-mono bg-orange-500/10 text-orange-400 border border-orange-500/20">{status}</span>;
      default:
        return <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium font-mono bg-gray-500/10 text-gray-400 border border-gray-500/20">{status}</span>;
    }
  };

  const isBoundaryOrSeg = filename.toLowerCase().includes('boundaries') || filename.toLowerCase().includes('segments');
  const isProjectTokenSegmentation = filename === '11_project_segments.json';
  const isEducationTokenSegmentation = filename === '4_education_segments.json';
  const isExperienceTokenSegmentation =
    filename === '8_experience_segments.json' || filename === '9_experience_segments.json';
  const isSkillsFields = filename === '7_skills_fields.json';
  const isHeadings = filename.toLowerCase().includes('headings');
  const isSectionLabels = filename.toLowerCase() === '3_section_labels.json';

  if (comparisonRows.length > 0 && stats) {
    return (
      <div className="space-y-3 my-2">
        {isProjectTokenSegmentation && (
          <div className="text-[10px] text-[var(--text-secondary)] space-y-1 max-w-3xl border border-[var(--border)] rounded-lg p-3 bg-[var(--bg-elevated)]">
            <p>
              <span className="font-semibold text-[var(--text-primary)]">Step 11 — Token segmentation</span>
              {' '}(training: <code>phase1_token_segmentation</code>). Groups tokens into phrase segments
              using <code>B-SEG</code> / <code>I-SEG</code> — same table as{' '}
              <code>phase1_token_segmentation/reports/minilm/per_resume/*.md</code>.
            </p>
            <p>
              Ground truth SEG is derived from MongoDB entity <code>bioLabel</code> (<code>B-PROJ_NAME</code>,{' '}
              <code>B-DESC</code>, … → <code>B-SEG</code>/<code>I-SEG</code>). Entry boundaries are step 12.
            </p>
          </div>
        )}
        {isEducationTokenSegmentation && (
          <div className="text-[10px] text-[var(--text-secondary)] space-y-1 max-w-3xl border border-[var(--border)] rounded-lg p-3 bg-[var(--bg-elevated)]">
            <p>
              <span className="font-semibold text-[var(--text-primary)]">Step 4 — Token segmentation</span>
              {' '}(training: <code>education/new_phase1_token_segmentation</code>). Groups tokens into phrase
              segments using <code>B-SEG</code> / <code>I-SEG</code>.
            </p>
            <p>
              GT SEG is derived from MongoDB <code>bioLabel</code> (<code>B-DEG</code>, <code>B-INST</code>, … →{' '}
              <code>B-SEG</code>/<code>I-SEG</code>). Entries are sliced via <code>educationEntryHeads</code>.
            </p>
          </div>
        )}
        {isExperienceTokenSegmentation && (
          <div className="text-[10px] text-[var(--text-secondary)] space-y-1 max-w-3xl border border-[var(--border)] rounded-lg p-3 bg-[var(--bg-elevated)]">
            <p>
              <span className="font-semibold text-[var(--text-primary)]">Step 8 — Token segmentation</span>
              {' '}(training: <code>experience/phase1_token_segmentation</code>). Groups tokens into phrase
              segments using <code>B-SEG</code> / <code>I-SEG</code> within each job entry.
            </p>
            <p>
              Entry boundaries come from step 9 (<code>9_experience_boundaries.json</code>, phase 2 divider).
              GT SEG is derived from MongoDB <code>bioLabel</code> (<code>B-ROLE</code>, <code>B-DESC</code>, … →{' '}
              <code>B-SEG</code>/<code>I-SEG</code>).
            </p>
          </div>
        )}
        {isSkillsFields && (
          <div className="text-[10px] text-[var(--text-secondary)] space-y-1 max-w-3xl border border-[var(--border)] rounded-lg p-3 bg-[var(--bg-elevated)]">
            <p>
              <span className="font-semibold text-[var(--text-primary)]">Step 7 — Skills token classification</span>
              {' '}(training: <code>skills/</code>). Direct 5-class BIO: <code>B-SKILL</code>, <code>I-SKILL</code>,{' '}
              <code>B-SKILL_TYPE</code>, <code>I-SKILL_TYPE</code>, <code>O</code>.
            </p>
            <p>
              Banner accuracy uses alphanumeric eval tokens only (same as{' '}
              <code>skills/reports/minilm/per_resume/*.md</code>). GT is MongoDB <code>bioLabel</code>.
            </p>
          </div>
        )}
        {/* Metric summary banner */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2 p-3.5 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] text-xs">
          <div>
            <div className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider font-mono">
              {filename.toLowerCase().includes('segments') || isSkillsFields ? 'Eval accuracy (alphanumeric)' : 'Accuracy'}
            </div>
            <div className={`text-base font-bold font-mono ${parseFloat(stats.accuracy) > 90 ? 'text-green-400' : 'text-yellow-400'}`}>{stats.accuracy}%</div>
          </div>
        <div>
          <div className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider font-mono">
            {filename.toLowerCase().includes('segments') || isSkillsFields ? 'Eval tokens' : 'Total Rows'}
          </div>
          <div className="text-base font-bold font-mono text-[var(--text-primary)]">
            {stats.evalTokens ?? stats.total}
          </div>
        </div>
          <div>
            <div className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider font-mono">Matches</div>
            <div className="text-base font-bold font-mono text-green-400">{stats.matches}</div>
          </div>
          <div>
            <div className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider font-mono">Mismatches</div>
            <div className="text-base font-bold font-mono text-red-400">{stats.mismatches}</div>
          </div>
          {!stats.headingsMode && !isSectionLabels && stats.notInDb > 0 && (
            <div>
              <div className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider font-mono">Not in DB</div>
              <div className="text-base font-bold font-mono text-purple-400">{stats.notInDb}</div>
            </div>
          )}
          {stats.notExtracted > 0 && (
            <div>
              <div className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider font-mono">Missed Extract</div>
              <div className="text-base font-bold font-mono text-orange-400">{stats.notExtracted}</div>
            </div>
          )}
        </div>




        {/* Filter controls toolbar */}
        <div className="flex flex-col sm:flex-row gap-2 justify-between items-stretch sm:items-center">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-[var(--text-secondary)]" />
            <input
              type="text"
              placeholder="Search token text, predictions..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 pr-3 py-1.5 w-full text-xs rounded border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)] font-mono"
            />
          </div>

          <div className="flex items-center gap-2 text-xs flex-wrap justify-end">
            {resumeId && (
              <button
                type="button"
                onClick={() => setGtRefreshTick((n) => n + 1)}
                disabled={gtRefreshing}
                className="inline-flex items-center gap-1 px-2 py-1 rounded border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-50"
                title="Reload ground truth from MongoDB (after viewer label edits)"
              >
                <RefreshCw className={`h-3 w-3 ${gtRefreshing ? 'animate-spin' : ''}`} />
                Refresh GT
              </button>
            )}
            <SlidersHorizontal className="h-3.5 w-3.5 text-[var(--text-secondary)]" />
            <span className="text-[var(--text-secondary)]">Show:</span>
            <div className="inline-flex rounded border border-[var(--border)] overflow-hidden">
              <button
                onClick={() => setStatusFilter('ALL')}
                className={`px-2.5 py-1 text-[11px] font-medium border-r border-[var(--border)] last:border-r-0 transition-colors ${
                  statusFilter === 'ALL'
                    ? 'bg-[var(--accent)] text-white'
                    : 'bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                All
              </button>
              <button
                onClick={() => setStatusFilter('ERRORS')}
                className={`px-2.5 py-1 text-[11px] font-medium border-r border-[var(--border)] last:border-r-0 transition-colors ${
                  statusFilter === 'ERRORS'
                    ? 'bg-red-600 text-white'
                    : 'bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                Errors Only
              </button>
              <button
                onClick={() => setStatusFilter('MATCHES')}
                className={`px-2.5 py-1 text-[11px] font-medium last:border-r-0 transition-colors ${
                  statusFilter === 'MATCHES'
                    ? 'bg-green-600 text-white'
                    : 'bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                Matches
              </button>
            </div>
          </div>
        </div>

        {/* Scrollable table container */}
        <div className="overflow-x-auto border border-[var(--border)] rounded-lg max-h-[60vh] bg-[var(--bg-elevated)]">
          <table className="w-full text-left border-collapse text-xs">
            <thead className="bg-[var(--bg-elevated)] sticky top-0 border-b border-[var(--border)] z-10 font-mono text-[10px] uppercase tracking-wider text-[var(--text-secondary)]">
              <tr>
                <th className="p-2.5 font-semibold">Status</th>
                {!isSectionLabels && <th className="p-2.5 font-semibold text-center">{isHeadings ? 'P/L' : 'P/L/T'}</th>}
                {!isHeadings && !isSectionLabels && <th className="p-2.5 font-semibold">Coords</th>}
                <th className="p-2.5 font-semibold">{isSectionLabels ? 'Section Heading' : isHeadings ? 'Extracted Heading' : 'Extracted Word'}</th>
                {isSectionLabels ? (
                  <>
                    <th className="p-2.5 font-semibold">Inferred Section</th>
                    <th className="p-2.5 font-semibold">GT Section</th>
                    <th className="p-2.5 font-semibold">Confidence &amp; Tokens</th>
                  </>
                ) : (
                  <>
                    <th className="p-2.5 font-semibold">{isHeadings ? 'Pipeline GT Heading' : 'DB/GT Word'}</th>
                    {isProjectTokenSegmentation && (
                      <th className="p-2.5 font-semibold">Mongo BIO</th>
                    )}
                    <th className="p-2.5 font-semibold">{isHeadings ? 'Source' : 'Prediction'}</th>
                    {!isHeadings && <th className="p-2.5 font-semibold">Ground Truth</th>}
                    {!isHeadings && <th className="p-2.5 font-semibold">DB Section</th>}
                  </>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border)] font-mono">
              {filteredRows.length > 0 ? (
                filteredRows.map((row, idx) => (
                  <tr key={idx} className="hover:bg-[var(--border)]/10 transition-colors">
                    <td className="p-2.5 whitespace-nowrap">{renderStatusBadge(row.status)}</td>
                    {!isSectionLabels && (
                      <td className="p-2.5 text-center text-[var(--text-secondary)] font-mono whitespace-nowrap">
                        {isHeadings ? `P${row.page} L${row.lineIndex}` : `P${row.page} L${row.lineIndex} T${row.tokenIndex}`}
                      </td>
                    )}
                    {!isHeadings && !isSectionLabels && <td className="p-2.5 text-[var(--text-secondary)] text-[10px] whitespace-nowrap font-mono">{row.coords}</td>}
                    <td className={`p-2.5 font-semibold font-sans ${row.token === null ? 'text-red-400 italic' : 'text-[var(--text-primary)]'}`}>
                      {row.token === null ? 'Missing' : row.token}
                    </td>
                    {isProjectTokenSegmentation && (
                      <td className="p-2.5 text-[var(--text-secondary)] text-[10px]">{row.mongoBioLabel ?? '—'}</td>
                    )}
                    {isSectionLabels ? (
                      <>
                        <td className="p-2.5">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium font-mono ${
                            row.infLabel === 'PERSONAL' ? 'bg-sky-500/10 text-sky-400 border border-sky-500/20' :
                            row.infLabel === 'SKILLS' ? 'bg-teal-500/10 text-teal-400 border border-teal-500/20' :
                            row.infLabel === 'EXPERIENCE' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' :
                            row.infLabel === 'EDUCATION' ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20' :
                            row.infLabel === 'PROJECT' ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20' :
                            row.infLabel === 'OTHER' ? 'bg-orange-500/10 text-orange-400 border border-orange-500/20' :
                            'bg-gray-500/10 text-gray-400'
                          }`}>{row.infLabel}</span>
                        </td>
                        <td className="p-2.5">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium font-mono ${
                            row.gtLabel === 'PERSONAL' ? 'bg-sky-500/10 text-sky-400 border border-sky-500/20' :
                            row.gtLabel === 'SKILLS' ? 'bg-teal-500/10 text-teal-400 border border-teal-500/20' :
                            row.gtLabel === 'EXPERIENCE' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' :
                            row.gtLabel === 'EDUCATION' ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20' :
                            row.gtLabel === 'PROJECT' ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20' :
                            row.gtLabel === 'OTHER' ? 'bg-orange-500/10 text-orange-400 border border-orange-500/20' :
                            'bg-gray-500/10 text-gray-400'
                          }`}>{row.gtLabel}</span>
                        </td>
                        <td className="p-2.5 text-[var(--text-secondary)] text-[10px] font-mono">{row.coords}</td>
                      </>
                    ) : (
                      <td className={`p-2.5 font-semibold font-sans ${row.gtToken === null ? 'text-purple-400 italic' : 'text-[var(--text-primary)]'}`}>
                        {row.gtToken === null ? 'None' : row.gtToken}
                      </td>
                    )}
                    {!isSectionLabels && (
                      <td className="p-2.5">
                        {isHeadings ? (
                          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium font-mono ${
                            row.infLabel.includes('heuristic') ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' :
                            row.infLabel.includes('minilm') ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20' :
                            'bg-gray-500/10 text-gray-400'
                          }`}>{row.infLabel}</span>
                        ) : (
                          <code className={`px-1 py-0.5 rounded text-[11px] ${
                            row.infLabel === 'O' || row.infLabel === '—'
                              ? 'bg-gray-500/10 text-gray-400'
                              : isBoundaryOrSeg
                                ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                                : 'bg-[var(--accent)]/10 text-[var(--accent)] border border-[var(--accent)]/20'
                          }`}>
                            {row.infLabel}
                          </code>
                        )}
                      </td>
                    )}
                    {!isHeadings && !isSectionLabels && (
                      <td className="p-2.5">
                        <code className={`px-1 py-0.5 rounded text-[11px] ${
                          row.gtLabel === 'O' || row.gtLabel === '—'
                            ? 'bg-gray-500/10 text-gray-400'
                            : isBoundaryOrSeg
                              ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                              : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        }`}>
                          {row.gtLabel}
                        </code>
                      </td>
                    )}
                    {!isHeadings && !isSectionLabels && <td className="p-2.5 text-[var(--text-secondary)] whitespace-nowrap">{row.section}</td>}
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={8} className="p-8 text-center text-[var(--text-secondary)] font-sans italic">
                    No matching comparison tokens found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // Inference-only preview (no GT or no comparison rows)
  if (data) {
    return <ArtifactPreviewView filename={filename} data={data} />;
  }

  return (
    <pre className="text-xs overflow-auto max-h-[50vh] p-3 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border)] font-mono">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}
