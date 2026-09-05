/**
 * The opening screen: what this session showed, in terms anyone can read.
 *
 * The job of this page is to land one idea before any chart appears -- that
 * reading lap times tells you about the situation, not about the tyre. It leads
 * with the comparison against the naive method because on real races that method
 * reports NEGATIVE degradation, which is a claim any reader can immediately
 * recognise as impossible.
 */

import { useEffect, useState } from 'react'
import {
  api,
  compoundColour,
  fixed,
  type SessionSummary,
  type RunRow,
} from '../lib/api'
import { DegradationCurves, QualityBreakdown, type CompoundCurve } from './charts'
import { Beam, CompoundChip, ErrorNote, Loading, Panel, Stat } from './primitives'
import { Explainer, Term } from './Explainer'

export function Overview({
  sessionId,
  onOpenExplain,
}: {
  sessionId: string
  onOpenExplain: () => void
}) {
  const [summary, setSummary] = useState<SessionSummary | null>(null)
  const [runs, setRuns] = useState<RunRow[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    setSummary(null)
    setError('')
    Promise.all([api.summary(sessionId), api.runs(sessionId)])
      .then(([s, r]) => {
        setSummary(s)
        setRuns(r)
      })
      .catch((e) => setError(String(e.message ?? e)))
  }, [sessionId])

  if (error) return <ErrorNote error={error} />
  if (!summary) return <Loading what="this session" />

  const compounds = Object.entries(summary.compounds)
  const naiveNegative = compounds.filter(
    ([, c]) => c.naive_estimate != null && c.naive_estimate < 0,
  )
  const longest = runs[0]

  // The curve is the fitted rate extended over the ages that compound was
  // actually run to -- drawn from the summary and the run table, so it needs no
  // extra request.
  const curves: CompoundCurve[] = Object.entries(summary.compounds)
    .map(([compound, estimate]) => ({
      compound,
      rate: estimate.degradation_rate,
      sd: estimate.degradation_rate_sd,
      maxAge: Math.max(
        0,
        ...runs.filter((r) => r.compound === compound).map((r) => r.end_age),
      ),
    }))
    .filter((c) => c.maxAge >= 5)

  return (
    <div className="space-y-3">
      <Explainer id="overview" question="What is this, in one paragraph?">
        <p>
          A Formula 1 tyre gets slower as it wears. Teams need to know{' '}
          <em>how much</em> slower, per lap, so they can decide when to pit. The
          obvious way to measure it is to watch lap times climb — and that
          obvious way is wrong.
        </p>
        <p>
          Lap times move for several reasons at once. The car gets lighter as it
          burns fuel, so it speeds up. The track gains grip as rubber goes down,
          so everyone speeds up. Catch another car and you lose time to{' '}
          <Term word="dirty air" meaning="Turbulence behind another car, which removes downforce from the car following." />
          . <strong>TyreMind separates these, and reports only the tyre.</strong>
        </p>
      </Explainer>

      {naiveNegative.length > 0 && (
        <Panel title="Why the obvious method fails" aside="this session, real data">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center">
            <div className="lg:w-[46%]">
              <p className="mb-3 max-w-[52ch] text-[13px] leading-relaxed text-ink">
                Fit a straight line through lap time against tyre age — the
                standard approach — and on this race it reports that the tyres got{' '}
                <strong className="text-alert">faster</strong> the longer they ran.
              </p>
              <p className="max-w-[52ch] text-[12px] leading-relaxed text-ink-dim">
                That is not a subtle error. It happens because fuel burn-off makes
                the car quicker by about 0.08 s a lap, which is larger than the
                tyre's degradation, so the tyre effect is buried under it and comes
                out with the wrong sign.
              </p>
            </div>

            <div className="flex-1 space-y-3">
              {compounds.map(([compound, estimate]) => (
                <div key={compound} className="border border-line bg-raised/40 p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <CompoundChip compound={compound} />
                    <span className="text-[10px] text-ink-faint">
                      {estimate.laps} laps
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="text-[10px] text-ink-faint">
                        Lap time vs tyre age
                      </div>
                      <div
                        className="num text-[19px] leading-tight"
                        style={{
                          color:
                            (estimate.naive_estimate ?? 0) < 0
                              ? 'var(--color-alert)'
                              : 'var(--color-ink-dim)',
                        }}
                      >
                        {fixed(estimate.naive_estimate)}
                      </div>
                      <div className="text-[10px] text-ink-faint">
                        {(estimate.naive_estimate ?? 0) < 0
                          ? 'tyre getting faster — impossible'
                          : 's/lap'}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-ink-faint">TyreMind</div>
                      <div
                        className="num text-[19px] leading-tight"
                        style={{ color: compoundColour(compound) }}
                      >
                        {fixed(estimate.degradation_rate)}
                      </div>
                      <div className="text-[10px] text-ink-faint">
                        ± {estimate.degradation_rate_sd.toFixed(3)} s/lap
                      </div>
                    </div>
                  </div>
                  <div className="mt-2">
                    <Beam
                      mean={estimate.degradation_rate}
                      sd={estimate.degradation_rate_sd}
                      domain={[-0.08, 0.28]}
                      colour={compoundColour(compound)}
                      zero
                      height={12}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={onOpenExplain}
            className="mt-4 border border-alert px-3 py-1.5 text-[12px] text-alert transition-colors hover:bg-alert/10"
          >
            See where the lap time actually went
          </button>
        </Panel>
      )}

      {curves.length > 0 && runs.length > 0 && (
        <Panel
          title="What the model actually estimates"
          aside="one curve per compound, with its uncertainty"
        >
          <div className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
            <DegradationCurves curves={curves} />
            <div className="space-y-3 text-[12.5px] leading-relaxed text-ink-dim">
              <p>
                Each line is how much performance a set of that compound has lost
                after a given number of laps, once fuel, track and traffic are
                taken out. The shaded band is the 95% range.
              </p>
              <p>
                Two things to look for. <strong className="text-ink">Is it
                straight?</strong> A curve that steepens is a tyre falling off its
                cliff. <strong className="text-ink">Where does the band flare?</strong>{' '}
                That is where the session ran out of laps on that compound, and the
                model is telling you it no longer has evidence.
              </p>
              <p className="text-[11.5px] text-ink-faint">
                This is the model's fitted estimate, not an average of the raw
                laps &mdash; the confounders have already been removed. Each line
                stops at the oldest tyre age that compound actually reached in
                this session, because past that point there is nothing to fit.
              </p>
            </div>
          </div>
        </Panel>
      )}

      <div className="grid gap-3 lg:grid-cols-[1.15fr_1fr]">
        <Panel title="What moved the lap times" aside="estimated for this session">
          <div className="space-y-4">
            <Effect
              label="Fuel burn-off"
              value={`${fixed(summary.confounders.fuel_slope.mean)} s/lap`}
              colour="var(--color-fuel)"
              plain="The car is heavy at the start and light at the end. Every lap it burns fuel, gets lighter, and goes quicker."
              caveat="This one is set by physics, not measured from the data — fuel and tyre wear both change smoothly with laps, so nothing in the timing sheet can tell them apart."
            />
            <Effect
              label="Track evolution"
              value={`${fixed(summary.confounders.track_evolution.mean, 2)} s over the session`}
              colour="var(--color-track)"
              plain="Cars leave rubber on the racing line. That rubber adds grip, so the circuit itself gets faster as the session goes on."
              caveat="Also partly assumed. We model it as a curve that flattens off, because rubber build-up genuinely saturates."
            />
            <Effect
              label="Traffic"
              value={`${fixed(summary.confounders.traffic.mean, 2)} s when worst`}
              colour="var(--color-traffic)"
              plain="Following another car costs downforce and therefore lap time. We work out who was behind whom from when each car started its lap."
              caveat="This one IS measured from the data, because traffic comes and goes independently of how old the tyre is."
            />
          </div>
        </Panel>

        <div className="space-y-3">
          <Panel title="This session" aside={`quality ${(summary.quality.quality_score ?? 0).toFixed(0)}/100`}>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-2">
              <Stat label="Laps analysed" value={String(summary.n_laps)} />
              <Stat label="Cars" value={String(summary.n_drivers)} />
              <Stat label="Tyre sets" value={String(summary.n_runs)} />
              <Stat
                label="Longest stint"
                value={longest ? String(longest.laps) : '—'}
                unit="laps"
              />
            </div>

            {summary.quality.exclusions && (
              <div className="mt-4 border-t border-line pt-3">
                <div className="mb-1 text-[11px] text-ink-faint">
                  Laps thrown out before analysis, and why
                </div>
                <QualityBreakdown
                  retained={summary.n_laps}
                  exclusions={summary.quality.exclusions}
                  labels={EXCLUSION_PLAIN}
                />
                <p className="mt-1 max-w-[44ch] text-[11px] leading-relaxed text-ink-faint">
                  Pit laps, safety-car laps and scruffy laps say nothing about the
                  tyre and would corrupt the estimate, so they are removed and
                  counted rather than quietly dropped.
                </p>
              </div>
            )}
          </Panel>

          <Panel title="How sure is it?" aside="model diagnostics">
            <div className="grid grid-cols-2 gap-4">
              <Stat
                label="Unexplained lap-time scatter"
                value={summary.diagnostics.observation_noise_sd.toFixed(3)}
                unit="s"
                hint="driver variation the model does not try to explain"
              />
              <Stat
                label="Quantities tracked"
                value={String(summary.diagnostics.n_states)}
                hint="one per car, per tyre set, plus the shared effects"
              />
            </div>
            <p className="mt-3 max-w-[46ch] text-[11.5px] leading-relaxed text-ink-dim">
              Every number on this screen carries a{' '}
              <Term
                word="credible interval"
                meaning="A range the true value probably falls in, given the data and the model's assumptions."
              />
              . The bars are drawn to scale — a wide bar means the model is genuinely
              unsure, and that is information rather than a defect.
            </p>
          </Panel>
        </div>
      </div>
    </div>
  )
}

function Effect({
  label,
  value,
  colour,
  plain,
  caveat,
}: {
  label: string
  value: string
  colour: string
  plain: string
  caveat: string
}) {
  return (
    <div className="border-l-2 pl-3.5" style={{ borderColor: colour }}>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-[13px] font-medium text-ink">{label}</span>
        <span className="num text-[13px] text-ink">{value}</span>
      </div>
      <p className="mt-1 max-w-[56ch] text-[12px] leading-relaxed text-ink-dim">{plain}</p>
      <p className="mt-1 max-w-[56ch] text-[11px] leading-relaxed text-ink-faint">{caveat}</p>
    </div>
  )
}

const EXCLUSION_PLAIN: Record<string, string> = {
  no_lap_time: 'No timed lap completed',
  pit_in_out_lap: 'Entering or leaving the pit lane',
  flagged_inaccurate: 'Timing data flagged unreliable',
  unknown_compound: 'Tyre compound not recorded',
  wet_compound: 'Wet-weather tyre (different physics)',
  slow_lap_safety_car_or_traffic: 'Far too slow — safety car, flags or heavy traffic',
  run_too_short: 'Stint too short to show a trend',
}
