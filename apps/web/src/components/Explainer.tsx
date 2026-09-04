/**
 * Plain-language framing for people who do not do this for a living.
 *
 * Every screen opens with one of these. The rule for the copy is that it must
 * be readable by someone who has never watched a race and never fitted a model,
 * without being wrong for someone who does both.
 *
 * Collapsed by default after first read, because an explanation that cannot be
 * dismissed becomes clutter for the second visit.
 */

import { useEffect, useState, type ReactNode } from 'react'

const STORAGE_PREFIX = 'tyremind.explainer.'

export function Explainer({
  id,
  question,
  children,
  defaultOpen = true,
}: {
  id: string
  question: string
  children: ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_PREFIX + id)
      if (stored !== null) setOpen(stored === 'open')
    } catch {
      /* private browsing, or storage disabled -- the default is fine */
    }
  }, [id])

  const toggle = () => {
    const next = !open
    setOpen(next)
    try {
      localStorage.setItem(STORAGE_PREFIX + id, next ? 'open' : 'closed')
    } catch {
      /* nothing to recover from; the panel still works this session */
    }
  }

  return (
    <div className="border-l-2 border-alert/60 bg-alert/[0.04]">
      <button
        onClick={toggle}
        className="flex w-full items-center gap-2 px-3.5 py-2 text-left"
        aria-expanded={open}
      >
        <span
          className="text-[10px] text-alert transition-transform"
          style={{ transform: open ? 'rotate(90deg)' : 'none' }}
          aria-hidden
        >
          ▶
        </span>
        <span className="text-[12px] font-semibold text-alert">{question}</span>
      </button>
      {open && (
        <div className="space-y-2 px-3.5 pb-3 pl-8 text-[12.5px] leading-relaxed text-ink-dim [&_strong]:text-ink">
          {children}
        </div>
      )}
    </div>
  )
}

/** Inline definition for a term of art, revealed on hover or focus. */
export function Term({ word, meaning }: { word: string; meaning: string }) {
  return (
    <span
      tabIndex={0}
      title={meaning}
      className="cursor-help border-b border-dotted border-ink-faint text-ink focus:outline-none focus-visible:border-alert"
    >
      {word}
    </span>
  )
}

/** The glossary, kept in one place so a term never means two things. */
export const GLOSSARY: Record<string, string> = {
  degradation:
    'How much slower a tyre gets with each lap it does, in seconds per lap. A rate, not a total.',
  compound:
    'The rubber recipe. Softer compounds grip better and wear out faster; harder ones last longer and are slower.',
  stint: 'A continuous run on one set of tyres, between pit stops.',
  'fuel burn-off':
    'A car gets lighter as it burns fuel, so it gets faster. Worth about 0.08 seconds a lap.',
  'track evolution':
    'The circuit gets faster during a session as cars lay down rubber, which adds grip for everyone.',
  'dirty air':
    'Turbulence behind another car. It removes downforce from the car following, costing lap time.',
  'the cliff':
    'The point where a tyre stops degrading gradually and starts falling apart quickly.',
  confounder:
    'Something that changes the lap time without the tyre having changed at all.',
  posterior:
    'What the model believes after seeing the data, including how sure it is.',
  filtered:
    'An estimate using only laps up to now. What the pit wall could legitimately know at the time.',
  smoothed:
    'An estimate using the whole session, including laps that came later. What engineers know afterwards.',
  coverage:
    'How often the true value actually falls inside the interval the model claimed. Should be about 95%.',
  CRPS: 'A score for a whole predicted distribution, not just its centre. Lower is better.',
}
