import React from 'react';

interface FieldRowProps {
  fieldKey: string;
  value: string;
}

const URL_SPLIT_RE = /(https?:\/\/[^\s]+)/g;

/** Wrap only URL substrings in anchors, leaving surrounding text as-is. */
function linkifyText(text: string): React.ReactNode[] {
  return text.split(URL_SPLIT_RE).map((part, idx) => {
    if (!/^https?:\/\//.test(part)) {
      return <React.Fragment key={idx}>{part}</React.Fragment>;
    }
    // Keep trailing sentence punctuation out of the link/href.
    const [, url = part, tail = ''] = part.match(/^(https?:\/\/.*?)([.,;:!?)\]}]*)$/) ?? [];
    return (
      <React.Fragment key={idx}>
        <a
          href={url}
          className="text-[var(--accent)] underline hover:opacity-80 transition-opacity break-all"
          target="_blank"
          rel="noopener noreferrer"
        >
          {url}
        </a>
        {tail}
      </React.Fragment>
    );
  });
}

export default function FieldRow({ fieldKey, value }: FieldRowProps) {
  if (!value) return null;

  const keyLower = fieldKey.toLowerCase();
  const isEmail = keyLower === 'email';
  const isPureUrl = /^https?:\/\/\S+$/.test(value.trim());
  const isLinkField =
    keyLower.startsWith('github') || keyLower.startsWith('linkedin') || keyLower.startsWith('link');
  // Only render the whole value as a single anchor for dedicated link fields or
  // when the value is nothing but a URL. Longer text (e.g. desc) is linkified inline.
  const isLink = isLinkField || isPureUrl;

  let href = '';
  if (isEmail) {
    href = `mailto:${value}`;
  } else if (isLink) {
    href = value.startsWith('http://') || value.startsWith('https://') ? value : `https://${value}`;
  }

  const renderValue = () => {
    if (isEmail || isLink) {
      return (
        <a
          href={href}
          className="text-[var(--accent)] underline hover:opacity-80 transition-opacity break-all"
          target={isLink ? '_blank' : undefined}
          rel={isLink ? 'noopener noreferrer' : undefined}
        >
          {value}
        </a>
      );
    }

    if (keyLower === 'desc') {
      return (
        <span className="text-[var(--text-primary)] whitespace-pre-wrap block mt-1">
          {linkifyText(value)}
        </span>
      );
    }

    return <span className="text-[var(--text-primary)]">{linkifyText(value)}</span>;
  };

  return (
    <div className={`text-sm py-1 ${keyLower === 'desc' ? 'block' : 'flex flex-wrap items-baseline gap-1.5'}`}>
      <span className="text-[var(--text-secondary)] font-mono">{keyLower}</span>
      {keyLower !== 'desc' && <span className="text-[var(--text-secondary)] font-mono">—</span>}
      {renderValue()}
    </div>
  );
}
