/**
 * Decomposition of one lap into the causes that produced it.
 *
 * Rendered as signed bars from a common zero rather than a stacked waterfall,
 * because the useful comparison is between the magnitudes of the causes, not
 * the running total. A race engineer wants to know which term is biggest and
 * whether the tyre one is significantly different from zero -- both of which a
 * shared baseline shows immediately and a staircase obscures.
 */

import type { Decomposition } from '../lib/api'
import { compoundColour, signed, TERM_COLOUR } from '../lib/api'
import { EstimateTag } from './primitives'

export function Waterfall({ decomposition }: { decomposition: Decomposition }) {
  const terms = [
    ...decomposition.contributions,
    {
      key: 'residual',
      label: 'Unexplained',
      seconds: decomposition.residual,
      sd: 0,
      ci95: [decomposition.residual, decomposition.residual] as [number, number],
      is_tyre: false,
    },
  ]

  const extent = Math.max(
    ...terms.map((t) => Math.abs(t.ci95[0])),
    ...terms.map((t) => Math.abs(t.ci95[1])),
    Math.abs(decomposition.observed_delta),
    0.1,
  )
  const domain: [number, number] = [-extent * 1.1, extent * 1.1]
  const toPct = (v: number) => ((v - domain[0]) / (domain[1] - domain[0])) * 100

  const tyreColour = compoundColour(decomposition.compound)
  const slower = decomposition.observed_delta > 0

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-end justify-between gap-4 border-b border-line pb-3">
        <div>
          <div className="text-[11px] text-ink-faint">
            {decomposition.driver} · lap {decomposition.session_lap} measured against lap{' '}
            {decomposition.reference_lap}
          </div>
          <div className="num mt-1 text-[30px] leading-none font-medium">
            {signed(decomposition.observed_delta)}
            <span className="ml-1.5 text-[12px] text-ink-faint">s</span>
          </div>
          <div className="mt-1 text-[11px] text-ink-dim">
            {slower ? 'slower' : 'faster'} than the reference lap
          </div>
        </div>
        <div className="text-right">
          <div className="text-[11px] text-ink-faint">of which the tyre is</div>
          <div className="num text-[30px] leading-none font-medium" style={{ color: tyreColour }}>
            {signed(decomposition.tyre_seconds)}
            <span className="ml-1.5 text-[12px] text-ink-faint">s</span>
          </div>
          <div className="mt-1 text-[11px] text-ink-dim">
            tyre age {decomposition.tyre_age.toFixed(0)} laps · {decomposition.compound}
          </div>
        </div>
      </div>

      <div className="space-y-2.5">
        {terms.map((term) => {
          const colour = term.is_tyre ? tyreColour : TERM_COLOUR[term.key] ?? 'var(--color-residual)'
          const lo = toPct(Math.min(term.ci95[0], term.ci95[1]))
          const hi = toPct(Math.max(term.ci95[0], term.ci95[1]))
          const barLo = toPct(Math.min(0, term.seconds))
          const barHi = toPct(Math.max(0, term.seconds))

          return (
            <div key={term.key} className="grid grid-cols-[120px_1fr_84px] items-center gap-3">
              <div
                className={`text-[12px] ${term.is_tyre ? 'font-semibold text-ink' : 'text-ink-dim'}`}
              >
                {term.label}
              </div>

              <div className="relative h-5 bg-raised">
                <div
                  className="absolute top-0 bottom-0 w-px bg-line-bright"
                  style={{ left: `${toPct(0)}%` }}
                />
                {term.sd > 0 && (
                  <div
                    className="beam absolute top-0 bottom-0"
                    style={{
                      left: `${lo}%`,
                      width: `${Math.max(hi - lo, 0.5)}%`,
                      ['--beam-color' as string]: colour,
                    }}
                  />
                )}
                <div
                  className="absolute top-1.5 bottom-1.5"
                  style={{
                    left: `${barLo}%`,
                    width: `${Math.max(barHi - barLo, 0.4)}%`,
                    background: colour,
                    opacity: term.is_tyre ? 1 : 0.75,
                  }}
                />
              </div>

              <div
                className={`num text-right text-[12.5px] ${
                  term.is_tyre ? 'text-ink' : 'text-ink-dim'
                }`}
              >
                {signed(term.seconds)}
              </div>
            </div>
          )
        })}
      </div>

      <KeyInsight decomposition={decomposition} />
    </div>
  )
}

/**
 * The sentence the whole product exists to be able to say.
 *
 * Deliberately conditional. When the car got faster overall, a "share of the
 * slowdown" is not a meaningful quantity, and printing one anyway would be the
 * exact overclaim the platform is built to avoid. That case is common and
 * interesting in its own right -- a stint where fuel masks a dying tyre.
 */
function KeyInsight({ decomposition }: { decomposition: Decomposition }) {
  const { observed_delta, tyre_seconds, confounder_seconds, tyre_share } = decomposition
  const gotFaster = observed_delta <= 0
  const tyreIsWorse = tyre_seconds > 0.05

  return (
    <div className="mt-5 border-l-2 pl-3.5" style={{ borderColor: 'var(--color-alert)' }}>
      <div className="mb-1 flex items-center gap-2">
        <span className="text-[11px] font-semibold tracking-tight text-alert">What this means</span>
        <EstimateTag />
      </div>
      <p className="max-w-[62ch] text-[13px] leading-relaxed text-ink">
        {gotFaster && tyreIsWorse ? (
          <>
            The car was <strong>{Math.abs(observed_delta).toFixed(2)} s faster</strong> than the
            reference lap, so the stopwatch says the tyre is fine. It is not. The tyre lost{' '}
            <strong style={{ color: compoundColour(decomposition.compound) }}>
              {tyre_seconds.toFixed(2)} s
            </strong>{' '}
            over this stint and the gain came from elsewhere &mdash; mostly fuel burn-off, worth{' '}
            {Math.abs(confounder_seconds).toFixed(2)} s. Reading pace alone would miss a degrading
            tyre completely.
          </>
        ) : Number.isFinite(tyre_share) ? (
          <>
            Only <strong>{(tyre_share * 100).toFixed(0)}%</strong> of this slowdown is the tyre. The
            remaining {((1 - tyre_share) * 100).toFixed(0)}% comes from conditions that changed
            around the car, not from the rubber on it.
          </>
        ) : (
          <>
            The car matched the reference lap, but the underlying terms did not cancel by accident.
            The tyre contributed {signed(tyre_seconds, 2)} s and everything else{' '}
            {signed(confounder_seconds, 2)} s.
          </>
        )}
      </p>
    </div>
  )
}
