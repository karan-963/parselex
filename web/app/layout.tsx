import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Parselex',
  description: 'Resume parsing inference pipeline',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen flex flex-col">
          <header className="h-14 bg-[var(--bg-surface)] border-b border-[var(--border)] flex items-center px-6 gap-4 sticky top-0 z-[100] backdrop-blur-sm shrink-0">
            <div className="flex items-center gap-2 text-[15px] font-bold tracking-[-0.3px] text-[var(--text-primary)]">
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <rect x="2" y="2" width="7" height="9" rx="1.5" fill="#6c8cff" opacity="0.8" />
                <rect x="11" y="2" width="7" height="5" rx="1.5" fill="#6c8cff" opacity="0.5" />
                <rect x="2" y="13" width="16" height="2" rx="1" fill="#6c8cff" opacity="0.4" />
                <rect x="2" y="17" width="10" height="2" rx="1" fill="#6c8cff" opacity="0.3" />
              </svg>
              Parselex
            </div>
          </header>
          <main className="flex-1">{children}</main>
        </div>
      </body>
    </html>
  );
}
