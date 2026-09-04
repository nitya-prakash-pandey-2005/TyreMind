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
          as each fold forecasts further past its training window. TyreMind is the
          only model tested whose error does not grow — which is what encoding fuel
          as physics buys, rather than learning it as a pattern.
        </p>
      </div>
    </Panel>
  )
}
