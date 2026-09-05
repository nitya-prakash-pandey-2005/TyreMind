/**
 * Pit strategy: thousands of simulated races, and the decision they support.
 *
 * The distribution plot is the point of this screen. A table of expected
 * finishing times hides the thing a strategist most needs to see -- whether two
 * options overlap so heavily that the difference between them is not real.
 *
 * Because the degradation rate is sampled from its posterior in every simulated
 * race rather than fixed at a point estimate, that overlap widens honestly when
 * the tyre estimate is uncertain.
 */

import { useEffect, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { useThemeColours } from '../lib/theme'
import {
  advanced,
  compoundColour,
  resolveColour,
  type PitWindow,
  type RunRow,
  type StrategyResult,
} from '../lib/api'
import { PitWindowChart } from './charts'
import { EstimateTag, ErrorNote, Loading, Panel, Stat } from './primitives'
import { Explainer } from './Explainer'

export function StrategyView({
  sessionId,
  runs,
  selected,
  onSelect,
}: {
  sessionId: string
  runs: RunRow[]
  selected: RunRow | null
  onSelect: (r: RunRow) => void
}) {
  const [lap, setLap] = useState<number | null>(null)
  const [result, setResult] = useState<StrategyResult | null>(null)
  const [regret, setRegret] = useState<{ regret_s: number } | null>(null)
  const [window, setWindow] = useState<PitWindow | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!selected) return
    setLap(Math.round((selected.first_lap + selected.last_lap) / 2))
  }, [selected])

  useEffect(() => {
    if (!selected || lap == null) return
    setBusy(true)
    setError('')
    advanced
      .strategy(sessionId, selected.driver, lap, 5000)
      .then(setResult)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false))
  }, [sessionId, selected, lap])

  // The sweep is a separate, slower call: it simulates every remaining lap, so
  // it must not hold up the headline recommendation.
  useEffect(() => {
    if (!selected || lap == null) return
    setWindow(null)
    advanced
      .pitWindow(sessionId, selected.driver, lap, 1200)
      .then(setWindow)
      .catch(() => setWindow(null))
  }, [sessionId, selected, lap])

  useEffect(() => {
    if (!selected || lap == null || !result) return
    const recommended = result.alternatives.find((a) => a.label === result.recommended)
    const pitLap = recommended?.pit_lap ?? lap + 1
    advanced
      .regret(sessionId, selected.driver, lap, pitLap, Math.min(pitLap + 5, selected.last_lap))
      .then(setRegret)
      .catch(() => setRegret(null))
  }, [sessionId, selected, lap, result])

  if (error) return <ErrorNote error={error} />
  if (!selected) return <Loading what="the session" />

  return (
    <div className="space-y-3">
      <Explainer id="strategy" question="What does this screen decide?">
        <p>
          When to pit. Stopping costs about twenty seconds in the pit lane, but
          fresh tyres are faster. Stop too early and you waste the tyre you were
          on; stop too late and you crawl round on a dead one.
        </p>
        <p>
          TyreMind plays the rest of the race out <strong>five thousand times</strong>{' '}
          for each option, drawing a different degradation rate each time from the
          range it believes. The chart below shows the spread of results. If two
          options overlap heavily, the model is telling you the choice does not
          really matter — which is as useful as being told it does.
        </p>
      </Explainer>

      <Panel title="Pick a car and a lap to decide from">
        <div className="mb-4 flex flex-wrap gap-1">
          {runs.slice(0, 12).map((run) => (
            <button
              key={run.run_id}
              onClick={() => onSelect(run)}
              className={`flex items-center gap-1.5 border px-2 py-1 text-[11px] transition-colors ${
                selected.run_id === run.run_id
                  ? 'border-alert text-ink'
                  : 'border-line text-ink-dim hover:border-line-bright'
              }`}
            >
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: compoundColour(run.compound) }}
              />
              <span className="num">{run.driver}</span>
              <span className="text-ink-faint">{run.laps}L</span>
            </button>
          ))}
        </div>

        {lap != null && (
          <label className="flex items-center gap-3 text-[11.5px] text-ink-dim">
            <span className="shrink-0">Deciding at lap</span>
            <input
              type="range"
              min={selected.first_lap}
              max={selected.last_lap}
              value={lap}
              onChange={(e) => setLap(Number(e.target.value))}
              className="max-w-md flex-1 accent-[var(--color-alert)]"
            />
            <span className="num w-8 text-ink">{lap}</span>
          </label>
        )}
      </Panel>

      {busy || !result ? (
        <Loading what="five thousand races" />
      ) : (
        <>
          <Panel
            title="Recommendation"
            aside={<EstimateTag>simulated, not observed</EstimateTag>}
          >
            <div className="flex flex-col gap-5 lg:flex-row">
              <div className="lg:w-[38%]">
                <div className="text-[11px] text-ink-faint">The model would</div>
                <div className="num mt-1 text-[30px] leading-none font-medium text-alert">
                  {result.recommended}
                </div>

                <div className="mt-4 grid grid-cols-2 gap-4">
                  <Stat
                    label="Wins how often"
                    value={`${(result.decision_confidence * 100).toFixed(0)}%`}
                    hint="against the next-best option"
                  />
                  <Stat
                    label="By"
                    value={result.margin_s.toFixed(1)}
                    unit="s"
                    hint={`over ${result.state.laps_remaining} remaining laps`}
                  />
                </div>

                {regret && regret.regret_s > 0.05 && (
                  <div className="mt-4 border-t border-line pt-3">
                    <div className="text-[11px] text-ink-faint">
                      Cost of stopping 5 laps later than recommended
                    </div>
                    <div className="num text-[22px] leading-none text-alert">
                      {regret.regret_s.toFixed(1)}
                      <span className="ml-1 text-[11px] text-ink-faint">s</span>
                    </div>
                    <p className="mt-1 max-w-[38ch] text-[11px] leading-relaxed text-ink-faint">
                      This is what model accuracy is worth in the only unit that
                      matters on a pit wall.
                    </p>
                  </div>
                )}
              </div>

              <div className="min-w-0 flex-1">
                <div className="mb-2 text-[11px] text-ink-faint">Why</div>
                <ul className="space-y-1.5">
                  {result.reasons.map((reason, i) => (
                    <li
                      key={i}
                      className="border-l-2 border-line pl-3 text-[12.5px] leading-relaxed text-ink-dim"
                    >
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </Panel>

          {window && window.sweep.length > 2 && (
            <Panel
              title="Every lap you could stop on"
              aside={`optimum lap ${window.optimum_lap}`}
            >
              <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
                <PitWindowChart
                  sweep={window.sweep}
                  optimum={window.optimum_lap}
                  window={window.window_within_1s}
                  stayOut={window.stay_out_expected_time}
                />
                <div className="space-y-3 text-[12.5px] leading-relaxed text-ink-dim">
                  <p>
                    The options table says which choice is best. This says{' '}
                    <strong className="text-ink">how much the choice matters</strong>,
                    which is the more useful question.
                  </p>
                  <StayOutNote window={window} />
                  {window.window_within_1s && (
                    <p>
                      Anywhere between lap{' '}
                      <span className="num text-ink">{window.window_within_1s[0]}</span> and{' '}
                      <span className="num text-ink">{window.window_within_1s[1]}</span>{' '}
                      costs less than a second against the optimum. That is a{' '}
                      {window.window_within_1s[1] - window.window_within_1s[0] + 1}-lap
                      window — comfortable room, not a knife edge.
                    </p>
                  )}
                  <p>
                    A steep curve means timing is critical. A flat one means the
                    decision does not really matter and the attention belongs
                    elsewhere. The dashed line is the bad case: where the curves
                    separate, one option is riskier than its average suggests.
                  </p>
                  <p className="text-[11.5px] text-ink-faint">
                    {window.sweep.length} candidate laps, {window.n_sims.toLocaleString()}{' '}
                    races each, all sharing random draws so the shape is signal
                    rather than simulation noise.
                  </p>
                </div>
              </div>
            </Panel>
          )}

          <div className="grid gap-3 lg:grid-cols-[1.25fr_1fr]">
            <Panel title="Five thousand races per option" aside="lower is faster">
              <DistributionChart result={result} />
            </Panel>

            <Panel title="Every option compared">
              <div className="space-y-1">
                <div className="grid grid-cols-[1fr_62px_58px_54px] gap-2 pb-1.5 text-[10px] text-ink-faint">
                  <span>option</span>
                  <span className="text-right">vs best</span>
                  <span className="text-right">bad case</span>
                  <span className="text-right">cliff</span>
                </div>
                {result.alternatives.map((option, i) => {
                  const best = result.alternatives[0]
                  const delta = option.expected_time - best.expected_time
                  return (
                    <div
                      key={option.label}
                      className="grid grid-cols-[1fr_62px_58px_54px] items-center gap-2 border-t border-line/60 py-1.5"
                    >
                      <span
                        className={`text-[12px] ${i === 0 ? 'font-medium text-alert' : 'text-ink-dim'}`}
                      >
                        {option.label}
                        {option.pit_lap && (
                          <span className="num ml-1.5 text-[10px] text-ink-faint">
                            L{option.pit_lap}
                          </span>
                        )}
                      </span>
                      <span className="num text-right text-[12px] text-ink">
                        {i === 0 ? '—' : `+${delta.toFixed(1)}s`}
                      </span>
                      <span className="num text-right text-[11.5px] text-ink-dim">
                        +{(option.downside - best.expected_time).toFixed(1)}s
                      </span>
                      <span
                        className="num text-right text-[11.5px]"
                        style={{
                          color:
                            option.ran_out_of_tyre > 0.5
                              ? 'var(--color-alert)'
                              : 'var(--color-ink-faint)',
                        }}
                      >
                        {(option.ran_out_of_tyre * 100).toFixed(0)}%
                      </span>
                    </div>
                  )
                })}
              </div>
              <p className="mt-3 max-w-[46ch] text-[11px] leading-relaxed text-ink-faint">
                <strong className="text-ink-dim">Bad case</strong> is the 90th
                percentile — how it goes when things do not fall your way.{' '}
                <strong className="text-ink-dim">Cliff</strong> is how often the tyre
                runs past the point where degradation accelerates.
              </p>
            </Panel>
          </div>
        </>
      )}
    </div>
  )
}

function DistributionChart({ result }: { result: StrategyResult }) {
  const c = useThemeColours()
  // Resolved to concrete values: ECharts draws to canvas and cannot read CSS
  // custom properties.
  const palette = [
    'var(--color-alert)',
    'var(--color-fuel)',
    'var(--color-traffic)',
    'var(--color-good)',
    'var(--color-residual)',
  ].map(resolveColour)

  const series = result.alternatives.slice(0, 5).map((option, i) => {
    const distribution = result.distributions[option.label]
    return {
      name: option.label,
      type: 'line',
      smooth: 0.4,
      symbol: 'none',
      // Race times are large numbers; showing them relative to the best option
      // makes the comparison legible instead of hiding it in the third digit.
      data: distribution
        ? distribution.centres.map((c, j) => [
            c - result.alternatives[0].expected_time,
            distribution.counts[j],
          ])
        : [],
      lineStyle: { width: i === 0 ? 2.4 : 1.4, color: palette[i % palette.length] },
      areaStyle: i === 0 ? { opacity: 0.16, color: palette[0] } : undefined,
      z: i === 0 ? 5 : 3,
    }
  })

  const option = {
    // Entry animation off: ECharts draws a line by expanding a clip path, and
    // these charts are re-initialised with `notMerge` whenever the session,
    // driver or theme changes. An entry animation interrupted by that re-init
    // leaves the clip frozen, so the series stops dead part-way across.
    animation: false,
    grid: { left: 42, right: 14, top: 30, bottom: 40 },
    legend: {
      top: 0,
      textStyle: { color: c.inkDim, fontSize: 10.5 },
      itemWidth: 14,
      itemHeight: 2,
    },
    xAxis: {
      type: 'value',
      name: 'seconds vs the best option',
      nameLocation: 'middle',
      nameGap: 24,
      nameTextStyle: { color: c.inkFaint, fontSize: 10.5 },
      axisLine: { lineStyle: { color: c.line } },
      axisLabel: { color: c.inkFaint, fontSize: 10.5, formatter: (v: number) => v.toFixed(0) },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: 'races',
      nameTextStyle: { color: c.inkFaint, fontSize: 10.5 },
      axisLine: { show: false },
      axisLabel: { color: c.inkFaint, fontSize: 10.5 },
      splitLine: { lineStyle: { color: c.raised } },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: c.surface,
      borderColor: c.line,
      textStyle: { color: c.ink, fontSize: 11.5 },
    },
    series,
  }

  return (
    <>
      <ReactECharts option={option} style={{ height: 300 }} notMerge />
      <p className="mt-2 max-w-[64ch] text-[11.5px] leading-relaxed text-ink-dim">
        Each curve is five thousand simulated races. Where curves overlap, the two
        strategies genuinely cannot be separated — the model is saying the choice is
        close, not hedging. The spread comes from real uncertainty about the tyre,
        because a different degradation rate is drawn from the posterior for every
        simulated race.
      </p>
    </>
  )
}

/**
 * The dotted reference line needs a sentence when it is the winning option.
 *
 * The sweep answers "which lap should I stop on"; not stopping is a different
 * question, and on a long final stint it is frequently the better answer. Saying
 * so explicitly stops the chart's own optimum from reading as a recommendation.
 */
function StayOutNote({ window: w }: { window: PitWindow }) {
  const bestStop = Math.min(...w.sweep.map((r) => r.expected_time))
  const margin = bestStop - w.stay_out_expected_time
  if (margin <= 0) return null
  return (
    <p>
      Every one of these stops loses to not stopping at all. The dotted line is
      staying out, and the best lap to pit on still costs{' '}
      <span className="num text-ink">{margin.toFixed(1)} s</span> against it. The
      curve below it is about which stop is least bad, not about whether to make
      one.
    </p>
  )
}
