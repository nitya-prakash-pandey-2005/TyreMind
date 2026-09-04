/**
 * Shared display primitives.
 *
 * The important one is `Beam`. Every estimate in TyreMind is drawn as an
 * interval rather than printed as a number with a plus-or-minus after it,
 * because the spread is usually the more decision-relevant half. A pit wall
 * reading "0.11 s/lap" acts differently from one reading "0.11, and it could
 * be anywhere from 0.07 to 0.16" -- so the interval is the primary mark and the
 * number annotates it, not the other way round.
 */

import type { ReactNode } from 'react'

export function Panel({
  title,
  aside,
  children,
  className = '',
}: {
  title?: string
  aside?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`bg-surface border border-line ${className}`}>
      {title && (
        <header className="flex items-baseline justify-between gap-4 border-b border-line px-4 py-2.5">
          <h2 className="text-[13px] font-semibold tracking-tight text-ink">{title}</h2>
          {aside && <div className="text-[11px] text-ink-faint">{aside}</div>}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}

/**
 * A posterior drawn to scale.
 *
 * @param mean Point estimate.
 * @param sd Posterior standard deviation. The beam spans the 95% interval.
 * @param domain Value range the track represents.
 * @param colour CSS colour for the beam and its tick.
 * @param zero Draw a reference line at zero, for signed quantities where the
 *   sign carries meaning (a degradation rate that could be negative is a
 *   different situation from one that definitely is not).
 */
export function Beam({
  mean,
  sd,
  domain,
  colour,
  zero = false,
  height = 18,
}: {
  mean: number
  sd: number
  domain: [number, number]
  colour: string
  zero?: boolean
  height?: number
}) {
  const [lo, hi] = domain
  const span = hi - lo || 1
  const pct = (v: number) => ((v - lo) / span) * 100

  const left = Math.max(0, pct(mean - 1.96 * sd))
  const right = Math.min(100, pct(mean + 1.96 * sd))
  const width = Math.max(right - left, 0.6)
  const meanPct = Math.min(Math.max(pct(mean), 0), 100)

  return (
    <div className="relative w-full bg-raised" style={{ height }}>
      {zero && lo < 0 && hi > 0 && (
        <div
          className="absolute top-0 bottom-0 w-px bg-line-bright"
          style={{ left: `${pct(0)}%` }}
        />
      )}
      <div
        className="beam absolute top-0 bottom-0"
        style={{ left: `${left}%`, width: `${width}%`, ['--beam-color' as string]: colour }}
      />
      <div
        className="absolute top-0 bottom-0 w-[2px]"
        style={{ left: `${meanPct}%`, background: colour }}
      />
    </div>
  )
}

/** A labelled figure. The label never competes with the number. */
export function Stat({
  label,
  value,
  unit,
  tone = 'default',
  hint,
}: {
  label: string
  value: string
  unit?: string
  tone?: 'default' | 'warm' | 'dim'
  hint?: string
}) {
  const colour =
    tone === 'warm' ? 'text-alert' : tone === 'dim' ? 'text-ink-dim' : 'text-ink'
  return (
    <div>
      <div className="text-[11px] text-ink-faint">{label}</div>
      <div className={`num mt-0.5 text-[22px] leading-none font-medium ${colour}`}>
        {value}
        {unit && <span className="ml-1 text-[11px] text-ink-faint">{unit}</span>}
      </div>
      {hint && <div className="mt-1 text-[11px] text-ink-faint">{hint}</div>}
    </div>
  )
}

/** Compound band, in Pirelli's own colours. */
export function CompoundChip({ compound }: { compound: string }) {
  const colour =
    { SOFT: 'var(--color-soft)', MEDIUM: 'var(--color-medium)', HARD: 'var(--color-hard)' }[
      compound?.toUpperCase()
    ] ?? 'var(--color-ink-dim)'
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] text-ink-dim">
      <span
        className="inline-block h-2.5 w-2.5 rounded-full border"
        style={{ borderColor: colour, background: `color-mix(in oklab, ${colour} 35%, transparent)` }}
      />
      {compound}
    </span>
  )
}

/**
 * Marks a number as inferred rather than observed.
 *
 * Used everywhere a counterfactual or projection is shown. The distinction
 * between "this lap was driven" and "this lap is what the model thinks would
 * have happened" is the one a strategy tool most needs to keep visible.
 */
export function EstimateTag({ children }: { children?: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1 border border-line px-1.5 py-0.5 text-[10px] text-ink-faint">
      <span className="inline-block h-1 w-1 rounded-full bg-ink-faint" />
      {children ?? 'model estimate'}
    </span>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="px-4 py-10 text-center text-[13px] text-ink-faint">{children}</div>
}

export function Loading({ what }: { what: string }) {
  return (
    <div className="flex items-center gap-2 px-4 py-10 text-[13px] text-ink-faint">
      <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-alert" />
      Fitting {what}…
    </div>
  )
}

export function ErrorNote({ error }: { error: string }) {
  return (
    <div className="border border-alert/40 bg-alert/5 px-4 py-3 text-[12px] text-ink-dim">
      <div className="mb-1 font-semibold text-alert">Could not load this</div>
      {error}
    </div>
  )
}
