import React, { useMemo } from 'react';
import { structureResume } from '../lib/structureResume';
import { extractSkillGroupsFromTokens, normalizeSkillList } from '../lib/skillsDisplay';
import { sectionStatKeyForResumeType, isVisibleSection } from '../lib/sectionPerformance';
import SectionCard from './SectionCard';
import FieldRow from './FieldRow';
import SkillBadgeGrid from './SkillBadgeGrid';
import type { PipelinePerformanceStats } from '../types';

interface StructuredResumeViewProps {
  data: any;
  /** Optional classified tokens for richer skills badge grouping. */
  tokens?: { tokens?: unknown[] } | unknown[] | null;
  performanceStats?: PipelinePerformanceStats | null;
}

export default function StructuredResumeView({ data, tokens, performanceStats }: StructuredResumeViewProps) {
  const sections = structureResume(data).filter((section) => section.type !== 'headings');
  const sectionStats = performanceStats?.sections;

  const tokenList = useMemo(() => {
    if (!tokens) return [];
    if (Array.isArray(tokens)) return tokens;
    if (typeof tokens === 'object' && tokens !== null && Array.isArray((tokens as { tokens?: unknown[] }).tokens)) {
      return (tokens as { tokens: unknown[] }).tokens;
    }
    return [];
  }, [tokens]);

  const skillGroupsFromTokens = useMemo(
    () => extractSkillGroupsFromTokens(tokenList as Parameters<typeof extractSkillGroupsFromTokens>[0]),
    [tokenList],
  );

  const statFor = (type: 'profile' | 'experience' | 'education' | 'projects' | 'skills') => {
    const key = sectionStatKeyForResumeType(type);
    const stat = key && sectionStats ? sectionStats[key] : undefined;
    return isVisibleSection(stat) ? stat : undefined;
  };

  if (!sections || sections.length === 0) {
    return (
      <div className="text-center p-8 border border-dashed border-[var(--border)] rounded-lg text-[var(--text-secondary)]">
        No structured resume data available.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {sections.map((section, idx) => {
        switch (section.type) {
          case 'profile':
            return (
              <SectionCard key={idx} title="Profile" stat={statFor('profile')}>
                {section.fields.map((f, i) => (
                  <FieldRow key={i} fieldKey={f.key} value={f.value} />
                ))}
              </SectionCard>
            );

          case 'summary':
            return (
              <SectionCard key={idx} title="Summary">
                <p className="text-sm text-[var(--text-primary)] whitespace-pre-wrap">
                  {section.text}
                </p>
              </SectionCard>
            );

          case 'experience':
            return (
              <SectionCard key={idx} title="Experience" stat={statFor('experience')}>
                {section.jobs.map((job, jobIdx) => (
                  <React.Fragment key={jobIdx}>
                    {jobIdx > 0 && (
                      <div className="border-t border-dotted border-[var(--border)] my-4" />
                    )}
                    <div className="text-center text-xs font-bold uppercase tracking-wide text-[var(--text-secondary)] my-3">
                      JOB {jobIdx + 1}
                    </div>
                    {job.fields.map((f, i) => (
                      <FieldRow key={i} fieldKey={f.key} value={f.value} />
                    ))}
                  </React.Fragment>
                ))}
              </SectionCard>
            );

          case 'education':
            return (
              <SectionCard key={idx} title="Education" stat={statFor('education')}>
                {section.entries.map((edu, eduIdx) => (
                  <React.Fragment key={eduIdx}>
                    {eduIdx > 0 && (
                      <div className="border-t border-dotted border-[var(--border)] my-4" />
                    )}
                    <div className="text-center text-xs font-bold uppercase tracking-wide text-[var(--text-secondary)] my-3">
                      EDUCATION {eduIdx + 1}
                    </div>
                    {edu.fields.map((f, i) => (
                      <FieldRow key={i} fieldKey={f.key} value={f.value} />
                    ))}
                  </React.Fragment>
                ))}
              </SectionCard>
            );

          case 'projects':
            return (
              <SectionCard key={idx} title="Projects" stat={statFor('projects')}>
                {section.entries.map((proj, projIdx) => (
                  <React.Fragment key={projIdx}>
                    {projIdx > 0 && (
                      <div className="border-t border-dotted border-[var(--border)] my-4" />
                    )}
                    <div className="text-center text-xs font-bold uppercase tracking-wide text-[var(--text-secondary)] my-3">
                      PROJECT {projIdx + 1}
                    </div>
                    {proj.fields.map((f, i) => (
                      <FieldRow key={i} fieldKey={f.key} value={f.value} />
                    ))}
                  </React.Fragment>
                ))}
              </SectionCard>
            );

          case 'skills': {
            const fallback = normalizeSkillList(section.items);
            return (
              <SectionCard key={idx} title="Skills" stat={statFor('skills')}>
                <SkillBadgeGrid groups={skillGroupsFromTokens} fallbackSkills={fallback} />
              </SectionCard>
            );
          }

          default:
            return null;
        }
      })}
    </div>
  );
}
