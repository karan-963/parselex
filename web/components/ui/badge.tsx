import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)]",
        secondary:
          "border-transparent bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)]",
        destructive:
          "border-transparent bg-red-500/15 text-red-400 border-red-500/30",
        outline: "border-[var(--border)] text-[var(--text-secondary)]",
        success:
          "border-transparent bg-[var(--approved-bg)] text-[var(--approved)] border-[rgba(34,197,94,0.3)]",
        warning:
          "border-transparent bg-[var(--pending-bg)] text-[var(--pending)] border-[rgba(245,158,11,0.3)]",
        muted:
          "border-transparent bg-[var(--skipped-bg)] text-[var(--skipped)]",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
