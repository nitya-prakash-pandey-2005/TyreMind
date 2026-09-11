/**
 * Method and evidence.
 *
 * Written for the reader who wants to know whether to believe any of this. It
 * shows the recorded experiment results as they are on disk -- nothing here is
 * hard-coded, and if an experiment has not been run the panel says so rather
 * than displaying a plausible number.
 *
 * The identifiability section is deliberately prominent. Two of the three
 * collinearities in this problem are resolved by assumption rather than by data,
 * and a tool that hid that would be easier to demo and worse to trust.
 */

import { useEffect, useState } from 'react'
import { api, fixed, signed, type SessionSummary } from '../lib/api'
import { Empty, Panel, Stat } from './primitives'
import { DegradationRegimes, PracticeVsRace, ReliabilityCurve } from './charts'

interface Recovery {
  n_seeds: number
  summary: {
    per_compound: Record<
      string,
      {
        true_rate: number
        ssm_mae: number
        naive_mae: number
        ssm_bias: number
        naive_bias: number
        interval_coverage_95: number
      }
    >
    overall: {
      ssm_mae: number
      naive_mae: number
      ssm_bias: number
      naive_bias: number
      error_reduction_pct: number
      interval_coverage_95: number
      mean_fit_seconds: number
    }
  }
}

interface CalibrationShape {
  n_races: number
  gaussian_diagonal_gap: number
  adaptive_diagonal_gap: number
  overconfident_rungs: string[]
  per_model: Record<
    string,
    {
      n_laps: number
      pit: { verdict?: string; tail_mass?: number; expected_tail_mass?: number }
      reliability: { levels: number[]; gaussian: number[]; adaptive: number[] }
    }
  >
}

interface CliffShapes {
  n_stints: number
  n_races: number
  regimes: Record<string, number>
  cliff_rate: number
  warmup_rate: number
  median_cliff_delta: number | null
  median_cliff_fraction: number | null
  forecast_by_regime: Record<
    string,
    { n: number; linear_mae: number; stick_mae: number; improvement_pct: number; linear_bias: number; wilcoxon_p: number }
  >
  stints: { regime: string; slope_before: number; delta: number; cliff_fraction: number }[]
}

interface ModelLadder {
  lap_time_prediction: {
    model: string
    crps: number
    mae: number
    coverage_95: number
    bias_drift: number
  }[]
  degradation_recovery: {
    model: string
    rate_mae: number
    rate_bias: number
    coverage: number
  }[]
  models_without_degradation_parameter: string[]
}

interface PracticeToRace {
  overall: {
    n_events: number
    n_comparisons: number
    mae: number
    naive_mae: number | null
    bias: number
    coverage_95: number
  }
  reports: {
    event: string
    comparisons: {
      compound: string
      predicted: number
      predicted_sd: number
      actual: number
      error: number
      covered_95: boolean
    }[]
  }[]
}

export function SciencePanel({ sessionId }: { sessionId: string }) {
  const [experiments, setExperiments] = useState<Record<string, unknown>>({})
  const [summary, setSummary] = useState<SessionSummary | null>(null)

  useEffect(() => {
    api.experiments().then(setExperiments).catch(() => undefined)
  }, [])
  useEffect(() => {
    api.summary(sessionId).then(setSummary).catch(() => undefined)
  }, [sessionId])

  const recovery = experiments['exp01_ground_truth_recovery'] as Recovery | undefined
  const transfer = experiments['exp03_practice_to_race'] as PracticeToRace | undefined
  const ladder = experiments['exp05_model_ladder'] as ModelLadder | undefined

  return (
    <div className="space-y-3">
      <Identifiability />

      <Panel
        title="Can it recover a degradation rate it was never shown?"
        aside={recovery ? `${recovery.n_seeds} synthetic sessions` : 'not yet run'}
      >
        {!recovery ? (
          <Empty>
            Run <span className="num">experiments/exp01_ground_truth_recovery.py</span> to
            populate this.
          </Empty>
        ) : (
          <>
            <p className="mb-4 max-w-[72ch] text-[12.5px] leading-relaxed text-ink-dim">
              Published tyre models are validated on lap-time prediction error. But a model can
              predict lap times almost perfectly while blaming the wrong cause &mdash; many wrong
              decompositions sum to the same right total. The only way to test attribution is to
              know the answer beforehand, so these sessions were generated with a hidden
              degradation rate and buried under realistic confounding.
            </p>

            <div className="mb-5 grid grid-cols-2 gap-5 sm:grid-cols-4">
              <Stat
                label="TyreMind error"
                value={recovery.summary.overall.ssm_mae.toFixed(4)}
                unit="s/lap"
                tone="warm"
              />
              <Stat
                label="Naive error"
                value={recovery.summary.overall.naive_mae.toFixed(4)}
                unit="s/lap"
                tone="dim"
              />
              <Stat
                label="Error reduction"
                value={`${recovery.summary.overall.error_reduction_pct.toFixed(1)}%`}
              />
              <Stat
                label="95% coverage"
                value={`${(recovery.summary.overall.interval_coverage_95 * 100).toFixed(0)}%`}
              />
            </div>

            <table className="w-full text-[12px]">
              <thead>
                <tr className="border-b border-line text-[10.5px] text-ink-faint">
                  <th className="py-1.5 text-left font-normal">compound</th>
                  <th className="text-right font-normal">true rate</th>
                  <th className="text-right font-normal">naive error</th>
                  <th className="text-right font-normal">TyreMind error</th>
                  <th className="text-right font-normal">coverage</th>
                </tr>
              </thead>
              <tbody className="num">
                {Object.entries(recovery.summary.per_compound).map(([compound, r]) => (
                  <tr key={compound} className="border-b border-line/50">
                    <td className="py-1.5 text-left font-sans">{compound}</td>
                    <td className="text-right text-ink-dim">{r.true_rate.toFixed(4)}</td>
                    <td className="text-right text-ink-dim">{r.naive_mae.toFixed(4)}</td>
                    <td className="text-right text-ink">{r.ssm_mae.toFixed(4)}</td>
                    <td className="text-right text-ink-dim">
                      {(r.interval_coverage_95 * 100).toFixed(0)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <p className="mt-3 max-w-[72ch] text-[11.5px] leading-relaxed text-ink-faint">
              The naive estimator&rsquo;s bias is{' '}
              <span className="num">{signed(recovery.summary.overall.naive_bias, 4)}</span> s/lap
              &mdash; almost exactly the fuel burn-off rate of 0.081 s/lap, and in the direction
              theory predicts. That is not a coincidence; it is the collinearity showing up as a
              measured quantity.
            </p>
          </>
        )}
      </Panel>

      <Panel
        title="Does a Friday curve predict Sunday?"
        aside={transfer ? `${transfer.overall.n_events} events, 2024` : 'not yet run'}
      >
        {!transfer ? (
          <Empty>
            Run <span className="num">experiments/exp03_practice_to_race.py</span> to populate
            this.
          </Empty>
        ) : (
          <>
            <p className="mb-4 max-w-[72ch] text-[12.5px] leading-relaxed text-ink-dim">
              Degradation is estimated from each event&rsquo;s practice session and scored against
              its race. No race data reaches the practice fit. Practice and race differ in fuel
              load, traffic density, track state and driving style at once, so this is genuine
              out-of-distribution transfer rather than a holdout split.
            </p>

            <div className="mb-5 grid grid-cols-2 gap-5 sm:grid-cols-4">
              <Stat
                label="TyreMind error"
                value={transfer.overall.mae.toFixed(4)}
                unit="s/lap"
                tone="warm"
              />
              <Stat
                label="Naive error"
                value={transfer.overall.naive_mae?.toFixed(4) ?? '—'}
                unit="s/lap"
                tone="dim"
              />
              <Stat
                label="Systematic bias"
                value={signed(transfer.overall.bias, 4)}
                unit="s/lap"
              />
              <Stat
                label="95% coverage"
                value={`${(transfer.overall.coverage_95 * 100).toFixed(0)}%`}
              />
            </div>

            <div className="mb-5 grid gap-5 xl:grid-cols-[1fr_1fr]">
              <div>
                <PracticeVsRace
                  points={transfer.reports.flatMap((r) =>
                    r.comparisons.map((c) => ({ ...c, event: r.event })),
                  )}
                  bias={transfer.overall.bias}
                />
              </div>
              <div className="space-y-3 text-[12.5px] leading-relaxed text-ink-dim">
                <p>
                  Each point is one compound at one event. The dashed diagonal is a
                  perfect prediction. The whisker is that prediction&rsquo;s own 95%
                  interval, and a ringed point is one the interval missed.
                </p>
                <p>
                  The points do not scatter around the diagonal &mdash; they sit
                  <strong className="text-ink"> above</strong> it, near the dotted
                  line. That is what makes the error{' '}
                  <strong className="text-ink">systematic rather than random</strong>,
                  and it is a far better position to be in: a consistent offset can be
                  corrected once its cause is understood, whereas scatter cannot be
                  corrected at all.
                </p>
                <p className="text-[11.5px] text-ink-faint">
                  Reported, not removed. Subtracting a bias measured on five events
                  would flatter these numbers and would not survive the sixth.
                </p>
              </div>
            </div>

            <table className="w-full text-[12px]">
              <thead>
                <tr className="border-b border-line text-[10.5px] text-ink-faint">
                  <th className="py-1.5 text-left font-normal">event</th>
                  <th className="text-left font-normal">compound</th>
                  <th className="text-right font-normal">from practice</th>
                  <th className="text-right font-normal">in the race</th>
                  <th className="text-right font-normal">error</th>
                </tr>
              </thead>
              <tbody className="num">
                {transfer.reports.flatMap((report) =>
                  report.comparisons.map((c) => (
                    <tr key={`${report.event}-${c.compound}`} className="border-b border-line/50">
                      <td className="py-1.5 text-left font-sans text-ink-dim">
                        {report.event.replace(' Grand Prix', '')}
                      </td>
                      <td className="text-left font-sans">{c.compound}</td>
                      <td className="text-right text-ink-dim">{fixed(c.predicted)}</td>
                      <td className="text-right text-ink-dim">{fixed(c.actual)}</td>
                      <td
                        className="text-right"
                        style={{
                          color: c.covered_95 ? 'var(--color-ink)' : 'var(--color-alert)',
                        }}
                      >
                        {signed(c.error)}
                      </td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>

            <div className="mt-4 border-l-2 border-alert pl-3.5">
              <div className="mb-1 text-[11px] font-semibold text-alert">
                An honest finding, not a clean win
              </div>
              <p className="max-w-[68ch] text-[12px] leading-relaxed text-ink-dim">
                The bias is systematic: practice over-predicts race degradation by{' '}
                <span className="num">{transfer.overall.bias.toFixed(3)}</span> s/lap, in most
                comparisons. The likely physical cause is that practice race-sim runs hold high
                fuel throughout while a race stint averages lower, putting more load through the
                tyre on Friday than on Sunday. A known, consistent bias is correctable; an
                unknown one is not, which is why it is reported here rather than tuned away.
              </p>
            </div>
          </>
        )}
      </Panel>

      {ladder && <ModelLadderPanel ladder={ladder} />}

      <CalibrationPanel />

      {summary && (
        <Panel title="This session's fit" aside="diagnostics">
          <div className="grid grid-cols-2 gap-5 sm:grid-cols-5">
            <Stat label="Laps used" value={String(summary.n_laps)} />
            <Stat label="Runs" value={String(summary.n_runs)} />
            <Stat label="Model states" value={String(summary.diagnostics.n_states)} />
            <Stat
              label="Residual noise"
              value={summary.diagnostics.observation_noise_sd.toFixed(3)}
              unit="s"
            />
            <Stat
              label="Data quality"
              value={`${(summary.quality.quality_score ?? 0).toFixed(0)}`}
              unit="/100"
            />
          </div>

          {summary.quality.exclusions && (
            <div className="mt-5 border-t border-line pt-4">
              <div className="mb-2 text-[11px] text-ink-faint">
                Laps removed before fitting, and why
              </div>
              <div className="flex flex-wrap gap-x-5 gap-y-1.5 text-[11.5px]">
                {Object.entries(summary.quality.exclusions).map(([reason, count]) => (
                  <span key={reason} className="text-ink-dim">
                    <span className="num text-ink">{count}</span>{' '}
                    {reason.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </div>
          )}
        </Panel>
      )}
    </div>
  )
}

/** The part most tools leave out. */
function Identifiability() {
  const items = [
    {
      title: 'Fuel against degradation',
      problem:
        'Within a run both are linear in laps completed. The car gets lighter and faster while the tyre gets slower, and from one run these cannot be separated at all.',
      fix: 'Pinned by a physical prior: 0.030 s/kg × 2.7 kg/lap. The prior is carried as state uncertainty, so it widens every interval rather than being assumed away.',
      resolved: 'assumption',
    },
    {
      title: 'Track evolution against degradation',
      problem:
        'Shifting every degradation rate by c and the track slope by −c leaves a difference that is constant within a run — exactly what the run intercept absorbs. Structurally indistinguishable.',
      fix: 'Track evolution is modelled as a saturating curve with an informative amplitude prior rather than a free random walk. Rubber deposition genuinely saturates, so this is also the more correct model.',
      resolved: 'assumption',
    },
    {
      title: 'Tyre age against session lap',
      problem:
        'They advance together within a run, so a single car cannot tell an ageing tyre from a changing session.',
      fix: 'Resolved by fitting the whole field at once. Cars change tyres on different laps, so at any session lap the grid spans a wide range of tyre ages. This one needs no prior — only the whole grid instead of one car.',
      resolved: 'data',
    },
  ]

  return (
    <Panel title="What can and cannot be identified" aside="the limits of the method">
      <p className="mb-4 max-w-[74ch] text-[12.5px] leading-relaxed text-ink-dim">
        Isolating degradation from a session is hard for a specific structural reason: several
        causes push lap time in the same monotone direction, so many wrong decompositions sum to
        the same right total. There are exactly three, and only one of them is resolved by
        evidence.
      </p>
      <div className="space-y-3.5">
        {items.map((item) => (
          <div
            key={item.title}
            className="border-l-2 pl-3.5"
            style={{
              borderColor:
                item.resolved === 'data' ? 'var(--color-good)' : 'var(--color-medium)',
            }}
          >
            <div className="flex items-baseline gap-2">
              <span className="text-[12.5px] font-medium text-ink">{item.title}</span>
              <span
                className="text-[10px]"
                style={{
                  color:
                    item.resolved === 'data' ? 'var(--color-good)' : 'var(--color-medium)',
                }}
              >
                {item.resolved === 'data' ? 'identified from data' : 'resolved by assumption'}
              </span>
            </div>
            <p className="mt-1 max-w-[74ch] text-[11.5px] leading-relaxed text-ink-faint">
              {item.problem}
            </p>
            <p className="mt-1 max-w-[74ch] text-[11.5px] leading-relaxed text-ink-dim">
              {item.fix}
            </p>
          </div>
        ))}
      </div>
    </Panel>
  )
}

/**
 * Two tables that disagree, which is the finding.
 *
 * A model can top the lap-time table while having nothing to say about tyres.
 * Showing only the table we win would misrepresent what was measured.
 */
function ModelLadderPanel({ ladder }: { ladder: ModelLadder }) {
  const noParameter = new Set(ladder.models_without_degradation_parameter)
  const bestLapTime = ladder.lap_time_prediction[0]?.model
  const bestRate = ladder.degradation_recovery[0]?.model

  return (
    <Panel title="Against every reasonable alternative" aside="identical chronological folds">
      <p className="mb-4 max-w-[74ch] text-[12.5px] leading-relaxed text-ink-dim">
        A state-space model is more complicated than a regression, so it has to
        earn that on the same data with the same validation. Two things are
        scored, and they disagree — which is the point.
      </p>

      <div className="grid gap-5 lg:grid-cols-2">
        <div>
          <div className="mb-2 text-[11px] text-ink-faint">
            Predicting lap times (4 real races)
          </div>
          <table className="w-full text-[11.5px]">
            <thead>
              <tr className="border-b border-line text-[10px] text-ink-faint">
                <th className="py-1.5 text-left font-normal">model</th>
                <th className="text-right font-normal">CRPS</th>
                <th className="text-right font-normal">cover</th>
                <th className="text-right font-normal">drift</th>
              </tr>
            </thead>
            <tbody className="num">
              {ladder.lap_time_prediction.map((row) => (
                <tr key={row.model} className="border-b border-line/50">
                  <td
                    className="py-1.5 text-left font-sans"
                    style={{
                      color:
                        row.model === bestLapTime ? 'var(--color-ink)' : 'var(--color-ink-dim)',
                    }}
                  >
                    {row.model}
                  </td>
                  <td className="text-right">{row.crps.toFixed(3)}</td>
                  <td className="text-right text-ink-dim">
                    {(row.coverage_95 * 100).toFixed(0)}%
                  </td>
                  <td
                    className="text-right"
                    style={{
                      color:
                        Math.abs(row.bias_drift) < 0.2
                          ? 'var(--color-good)'
                          : 'var(--color-ink-faint)',
                    }}
                  >
                    {signed(row.bias_drift, 2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <div className="mb-2 text-[11px] text-ink-faint">
            Recovering a known degradation rate (synthetic)
          </div>
          <table className="w-full text-[11.5px]">
            <thead>
              <tr className="border-b border-line text-[10px] text-ink-faint">
                <th className="py-1.5 text-left font-normal">model</th>
                <th className="text-right font-normal">error</th>
                <th className="text-right font-normal">bias</th>
                <th className="text-right font-normal">cover</th>
              </tr>
            </thead>
            <tbody className="num">
              {ladder.degradation_recovery.map((row) => (
                <tr key={row.model} className="border-b border-line/50">
                  <td
                    className="py-1.5 text-left font-sans"
                    style={{
                      color: row.model === bestRate ? 'var(--color-alert)' : 'var(--color-ink-dim)',
                    }}
                  >
                    {row.model}
                  </td>
                  <td className="text-right">{row.rate_mae.toFixed(4)}</td>
                  <td className="text-right text-ink-dim">{signed(row.rate_bias, 4)}</td>
                  <td className="text-right text-ink-dim">
                    {(row.coverage * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
              {[...noParameter].map((model) => (
                <tr key={model} className="border-b border-line/50">
                  <td className="py-1.5 text-left font-sans text-ink-faint">{model}</td>
                  <td colSpan={3} className="text-right text-[10.5px] text-ink-faint">
                    no degradation parameter
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-4 border-l-2 border-alert pl-3.5">
        <div className="mb-1 text-[11px] font-semibold text-alert">
          The best lap-time predictor cannot answer the question
        </div>
        <p className="max-w-[70ch] text-[12px] leading-relaxed text-ink-dim">
          <strong className="text-ink">{bestLapTime}</strong> predicts lap times
          better than we do. It has no parameter meaning &ldquo;degradation rate&rdquo;, so
          there is nothing to hand an engineer and nothing to carry from Friday to
          Sunday. It is also badly overconfident.
        </p>
        <p className="mt-1.5 max-w-[70ch] text-[12px] leading-relaxed text-ink-dim">
          <strong className="text-ink">Drift</strong> is how much a model&rsquo;s error grows
          as each fold forecasts further past its training window. TyreMind&rsquo;s
          shrinks the most of any usable rung while the lap-time leader&rsquo;s grows —
          which is what encoding fuel as physics buys, rather than learning it as a
          pattern. On four races we called it the <em>only</em> such rung; on twenty
          that was no longer true, and the claim is corrected rather than restated.
        </p>
      </div>
    </Panel>
  )
}


/**
 * Whether the intervals mean what they say, and what shape the curve really is.
 *
 * Two findings that only exist because a single headline number was not trusted.
 * Coverage at one nominal level cannot distinguish a calibrated model from one
 * tuned to look right at 95%, so the level is swept and the diagonal drawn. And
 * a changepoint fitted to a stint finds two physically opposite things — a tyre
 * giving up and a tyre coming to temperature — which an earlier analysis pooled,
 * reporting a "cliff" a third of the way through a stint.
 */
export function CalibrationPanel() {
  const [experiments, setExperiments] = useState<Record<string, unknown>>({})
  useEffect(() => {
    api.experiments().then(setExperiments).catch(() => undefined)
  }, [])

  const shape = experiments['exp16_calibration_shape'] as CalibrationShape | undefined
  const cliff = experiments['exp17_degradation_cliff'] as CliffShapes | undefined

  return (
    <div className="space-y-3">
      <Panel
        title="Does a 95% interval contain the answer 95% of the time?"
        aside={shape ? `${shape.n_races} races, every rung` : 'not yet run'}
      >
        {!shape ? (
          <Empty>
            Run <span className="num">experiments/exp16_calibration_shape.py</span> to
            populate this.
          </Empty>
        ) : (
          <>
            <p className="mb-4 max-w-[74ch] text-[12.5px] leading-relaxed text-ink-dim">
              Coverage quoted at one level is a weak check. A model can hit 95% exactly
              while being far too confident in the middle of its distribution and far too
              timid in the tails, and a single percentage would look perfect either way.
              Sweeping the nominal level turns the claim into a curve: a calibrated method
              traces the diagonal, and one tuned to look right at 95% does not.
            </p>

            <div className="mb-5 grid grid-cols-2 gap-5 sm:grid-cols-3">
              <Stat
                label="Distance from the diagonal, as reported"
                value={shape.gaussian_diagonal_gap.toFixed(3)}
                tone="dim"
              />
              <Stat
                label="Distance from the diagonal, calibrated"
                value={shape.adaptive_diagonal_gap.toFixed(3)}
                tone="warm"
              />
              <Stat
                label="Rungs whose PIT is U-shaped"
                value={`${shape.overconfident_rungs.length}/${Object.keys(shape.per_model).length}`}
              />
            </div>

            <ReliabilityCurve
              rows={Object.entries(shape.per_model).map(([model, m]) => ({
                model,
                levels: m.reliability.levels,
                gaussian: m.reliability.gaussian,
                adaptive: m.reliability.adaptive,
              }))}
            />

            <div className="mt-4 border-l-2 border-alert pl-3.5">
              <div className="mb-1 text-[11px] font-semibold text-alert">
                The shape names the fault, not just its size
              </div>
              <p className="max-w-[74ch] text-[12px] leading-relaxed text-ink-dim">
                A PIT histogram asks where the truth landed inside its own predicted
                distribution; under a correct distribution those values are uniform.
                Every rung comes out <strong className="text-ink">U-shaped</strong> — too
                much mass in the tails — which is overconfidence seen directly rather than
                inferred from one level. A four-race pilot showed the state-space model
                hump-shaped instead, the opposite fault; ten races showed that was a
                small-sample artefact.
              </p>
            </div>
          </>
        )}
      </Panel>

      <Panel
        title="Is the degradation curve a straight line?"
        aside={cliff ? `${cliff.n_stints} stints, ${cliff.n_races} races` : 'not yet run'}
      >
        {!cliff ? (
          <Empty>
            Run <span className="num">experiments/exp17_degradation_cliff.py</span> to
            populate this.
          </Empty>
        ) : (
          <>
            <p className="mb-4 max-w-[74ch] text-[12.5px] leading-relaxed text-ink-dim">
              A linear rate answers &ldquo;how fast is this tyre losing performance&rdquo;.
              It cannot answer &ldquo;how many laps have I got left&rdquo;, and the two are
              only the same question if the curve is a line. Fitting a broken stick to each
              de-confounded stint says how often it is not.
            </p>

            <DegradationRegimes
              rows={Object.entries(cliff.regimes).map(([regime, n]) => {
                const block = cliff.stints.filter((st) => st.regime === regime)
                const median = (xs: number[]) =>
                  xs.length ? xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)] : null
                return {
                  regime,
                  n,
                  share: n / cliff.n_stints,
                  slopeBefore: median(block.map((b) => b.slope_before)),
                  delta: regime === 'linear' ? null : median(block.map((b) => b.delta)),
                  position: regime === 'linear' ? null : median(block.map((b) => b.cliff_fraction)),
                }
              })}
            />

            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <div className="border-l-2 border-alert pl-3.5">
                <div className="mb-1 text-[11px] font-semibold text-alert">
                  A cliff is real, and arrives too late to act on
                </div>
                <p className="text-[12px] leading-relaxed text-ink-dim">
                  {Math.round(cliff.cliff_rate * 100)}% of stints end in a genuine cliff —
                  the tyre already degrading, then degrading{' '}
                  <span className="num text-ink">
                    {cliff.median_cliff_delta === null
                      ? ''
                      : `${signed(cliff.median_cliff_delta, 3)} s/lap`}
                  </span>{' '}
                  faster, at{' '}
                  {cliff.median_cliff_fraction === null
                    ? ''
                    : `${Math.round(cliff.median_cliff_fraction * 100)}%`}{' '}
                  through the stint. Fitting the opening 70% does not locate it, because it
                  has not happened yet. What it does establish is the direction of the
                  error: a straight line is systematically <em>optimistic</em> about the
                  laps that matter most.
                </p>
              </div>
              <div className="border-l-2 border-line pl-3.5">
                <div className="mb-1 text-[11px] font-semibold text-ink">
                  Warm-up is the opposite event, and was nearly missed
                </div>
                <p className="text-[12px] leading-relaxed text-ink-dim">
                  {Math.round(cliff.warmup_rate * 100)}% of stints show the tyre getting{' '}
                  <em>quicker</em> and then turning — coming to temperature, or a graining
                  phase clearing. It is a changepoint with the same arithmetic signature as
                  a cliff and the opposite meaning. Pooling the two, which the first pass
                  did, reports a &ldquo;cliff&rdquo; a third of the way through a stint.
                </p>
              </div>
            </div>
          </>
        )}
      </Panel>
    </div>
  )
}
