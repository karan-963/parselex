'use client';

import { Badge } from '@/components/ui/badge';
import type { SkillGroup } from '../lib/skillsDisplay';
import { cleanDisplayText } from '../lib/textClean';

interface Props {
  groups: SkillGroup[];
  /** Flat fallback when no grouped spans are available. */
  fallbackSkills?: string[];
}

function SkillBadges({ skills }: { skills: string[] }) {
  const cleaned = skills.map(cleanDisplayText).filter((s) => s.length > 0);
  return (
    <div className="flex flex-wrap gap-2">
      {cleaned.map((skill) => (
        <Badge
          key={skill}
          className="px-2.5 py-1 text-xs font-normal border border-[var(--border)] bg-[var(--bg-active)] text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
        >
          {skill}
        </Badge>
      ))}
    </div>
  );
}

export default function SkillBadgeGrid({ groups, fallbackSkills = [] }: Props) {
  const hasGroups = groups.some((g) => g.skills.length > 0);

  if (!hasGroups && fallbackSkills.length === 0) {
    return <p className="text-xs text-[var(--text-secondary)] italic">No skills detected.</p>;
  }

  if (!hasGroups) {
    return <SkillBadges skills={fallbackSkills} />;
  }

  return (
    <div className="space-y-4">
      {groups.map((group, idx) => (
        <div key={`${group.category ?? 'all'}-${idx}`} className="space-y-2">
          {group.category && cleanDisplayText(group.category) && (
            <div className="text-[11px] font-semibold text-[var(--text-primary)]">
              {cleanDisplayText(group.category)}
            </div>
          )}
          <SkillBadges skills={group.skills} />
        </div>
      ))}
    </div>
  );
}
