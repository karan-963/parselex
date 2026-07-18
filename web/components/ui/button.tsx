import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-40 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]",
  {
    variants: {
      variant: {
        default:
          "bg-[var(--accent)] text-white shadow-[0_0_16px_rgba(108,140,255,0.25)] hover:bg-[var(--accent-hover)] hover:shadow-[0_0_24px_rgba(108,140,255,0.35)]",
        destructive:
          "bg-red-500/12 text-red-400 border border-red-500/25 hover:bg-red-500/20",
        outline:
          "border border-[var(--border)] bg-transparent text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)] hover:border-[var(--border-strong)]",
        secondary:
          "bg-[var(--bg-elevated)] text-[var(--text-secondary)] border border-[var(--border)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]",
        ghost:
          "bg-transparent text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]",
        success:
          "bg-[var(--approved-bg)] text-[var(--approved)] border border-[rgba(34,197,94,0.3)] hover:bg-[rgba(34,197,94,0.25)]",
        warning:
          "bg-[var(--pending-bg)] text-[var(--pending)] border border-[rgba(245,158,11,0.3)] hover:bg-[rgba(245,158,11,0.25)]",
        link: "text-[var(--accent)] underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md gap-1.5 px-3 text-xs",
        lg: "h-10 rounded-lg px-6",
        icon: "size-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
