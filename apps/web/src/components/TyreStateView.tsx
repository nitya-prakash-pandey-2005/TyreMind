/**
 * The tyre digital twin: what the model currently believes about one set of tyres.
 *
 * Combines the four things a race engineer asks in sequence -- how worn is it,
 * how is it being worked, how long has it got, and what would help. The health
 * timeline sits underneath, so a single number is never shown without the
 * trajectory that produced it.
 */

import { useEffect, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import {
  advanced,
  api,
  compoundColour,
  compoundColourResolved,
  cornerEnergy,
  resolveColour,
  signed,
  type CornerEnergy,
  type HealthTimeline,
  type ProjectionResult,
  type RunRow,
  type Scenario,
  type TrustResult,
} from '../lib/api'
import { Beam, EstimateTag, ErrorNote, Loading, Panel, Stat } from './primitives'
import { Explainer } from './Explainer'
import { EVEN_SPLIT, TyreTwin } from './TyreTwin'

export function TyreStateView({
  sessionId,
  circuit,
  runs,
  selected,
  onSelect,
}: {
  sessionId: string
  circuit: string
  runs: RunRow[]
  selected: RunRow | null
  onSelect: (r: RunRow) => void
}) {
  const [timeline, setTimeline] = useState<HealthTimeline | null>(null)
  const [projection, setProjection] = useState<ProjectionResult | null>(null)
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [corners, setCorners] = useState<CornerEnergy | null>(null)
  const [trust, setTrust] = useState<TrustResult | null>(null)
  const [narration, setNarration] = useState<string>('')
  const [error, setError] = useState('')

  useEffect(() => {
    cornerEnergy(circuit).then(setCorners).catch(() => setCorners(null))
  }, [circuit])

  // Each panel loads independently rather than through Promise.all. The trust
  // endpoint refits the model under perturbed priors and takes several seconds
  // on a cold cache; bundling it with the rest left every panel blank until the
  // slowest one finished.
  useEffect(() => {
    if (!selected) return
    const lap = Math.round((selected.first_lap + selected.last_lap) / 2)

    setError('')
    setTimeline(null)
    setProjection(null)
    setScenarios([])
    setTrust(null)
    setNarration('')

    advanced
      .healthTimeline(sessionId, selected.driver, selected.run_id)
      .then(setTimeline)
      .catch((e) => setError(String(e.message ?? e)))

    api.projection(sessionId, selected.driver, lap, 18).then(setProjection).catch(() => undefined)
    api
      .counterfactual(sessionId, selected.driver, lap)
      .then((c) => setScenarios(c.scenarios))
      .catch(() => undefined)
    advanced
      .narrate(sessionId, selected.driver, lap)
      .then((n) => setNarration(n.projection.text))
      .catch(() => undefined)
    advanced
      .trust(sessionId, selected.compound, selected.end_age)
      .then(setTrust)
      .catch(() => undefined)
  }, [sessionId, selected])

  if (error) return <ErrorNote error={error} />
  if (!selected) return <Loading what="the session" />

  const latest = timeline?.rows.at(-1)

  return (
    <div className="space-y-3">
      <Explainer id="tyrestate" question="What is a tyre digital twin?">
        <p>
          A running estimate of the condition of one specific set of tyres,
          updated every lap. It is not a measurement — nothing on an F1 car reports
          tread depth to the outside world. It is what the model infers from how
          the car has been performing, with the confounders taken out.
        </p>
        <p>
          <strong>Health is a convention, not a reading.</strong> 100 means as new;
          0 means the set is 1.5 seconds a lap slower than a fresh one. The seconds
          are always shown next to it so you can check the conversion yourself.
        </p>
      </Explainer>

      <Panel title="Choose a set of tyres">
        <div className="flex flex-wrap gap-1">
          {runs.slice(0, 14).map((run) => (
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
      </Panel>

      <div className="grid gap-3 lg:grid-cols-[1fr_1.1fr]">
        <Panel
          title="Digital twin"
          aside={corners?.measured ? `measured at ${corners.circuit}` : 'loading split'}
        >
          <TyreTwin
            energy={corners?.corner_share ?? EVEN_SPLIT}
            health={latest?.health ?? null}
            performanceLost={latest?.level ?? null}
            compound={selected.compound}
            ageLaps={selected.end_age}
          />

          {corners && !corners.measured && (
            <p className="mt-3 border-l-2 border-medium pl-3 text-[11px] leading-relaxed text-ink-faint">
              Corner loading has not been computed for {corners.circuit}. An even
              split is shown as a placeholder — it is not a result.{' '}
              {corners.reason}
            </p>
          )}
          {corners?.measured && (
            <p className="mt-3 border-l-2 border-good pl-3 text-[11px] leading-relaxed text-ink-faint">
              Measured from {corners.n_laps} laps of position telemetry. The model
              was never told which way {corners.circuit} runs, and inferred{' '}
              <strong className="text-ink-dim">{corners.predicted_direction}</strong> from
              the loading alone — which matches the published circuit map. Peak
              lateral load {corners.peak_lateral_g?.toFixed(1)} g.
            </p>
          )}
        </Panel>

        <Panel title="How this set has aged" aside="lap by lap">
          {timeline ? (
            <HealthChart timeline={timeline} />
          ) : (
            <Loading what="the timeline" />
          )}
        </Panel>
      </div>

      <div className="grid gap-3 lg:grid-cols-[1fr_1fr]">
        {projection && (
          <Panel title="How much life is left" aside={<EstimateTag>projection</EstimateTag>}>
            <div className="mb-4 grid grid-cols-3 gap-4">
              <Stat label="Age now" value={projection.tyre_age.toFixed(0)} unit="laps" />
              <Stat
                label="Competitive laps left"
                value={projection.competitive_life_laps.toFixed(0)}
                tone="warm"
                hint={`between ${projection.competitive_life_lower.toFixed(0)} and ${projection.competitive_life_upper.toFixed(0)}`}
              />
              <Stat
                label="Threshold"
                value={String(projection.threshold_s)}
                unit="s/lap"
                tone="dim"
                hint="slower than fresh"
              />
            </div>

            {narration && (
              <p className="mb-4 border-l-2 border-alert pl-3 text-[12.5px] leading-relaxed text-ink">
                {narration}
              </p>
            )}

            <div className="space-y-1.5">
              <div className="grid grid-cols-[46px_1fr_46px_46px] gap-2 text-[10px] text-ink-faint">
                <span>ahead</span>
                <span>chance it is past the threshold</span>
                <span className="text-right">%</span>
                <span className="text-right">applies</span>
              </div>
              {[1, 3, 5, 8, 12, 17].map((h) => {
                const i = h - 1
                if (i >= projection.horizon.length) return null
                const p = projection.breach_probability[i]
                const applies = projection.applicability[i]
                return (
                  <div
                    key={h}
                    className="grid grid-cols-[46px_1fr_46px_46px] items-center gap-2 border-t border-line/60 py-1"
                  >
                    <span className="num text-[11.5px] text-ink-dim">+{h}</span>
                    <div className="h-3 bg-raised">
                      <div
                        className="h-full transition-[width]"
                        style={{
                          width: `${p * 100}%`,
                          background: compoundColour(projection.compound),
                          opacity: 0.5 + 0.5 * applies,
                        }}
                      />
                    </div>
                    <span className="num text-right text-[11.5px]">
                      {(p * 100).toFixed(0)}
                    </span>
                    <span
                      className="num text-right text-[10.5px]"
                      style={{
                        color: applies < 0.5 ? 'var(--color-alert)' : 'var(--color-ink-faint)',
                      }}
                    >
                      {(applies * 100).toFixed(0)}
                    </span>
                  </div>
                )
              })}
            </div>
            <p className="mt-3 max-w-[52ch] text-[11px] leading-relaxed text-ink-faint">
              The right-hand column falls once the projection reaches past the oldest
              tyre this session actually contains. Below 50% the model is
              extrapolating rather than reporting, and a cliff outside the observed
              range cannot be seen at all.
            </p>
          </Panel>
        )}

        <div className="space-y-3">
          <Panel title="What if" aside={<EstimateTag>never driven</EstimateTag>}>
            <div className="space-y-3.5">
              {scenarios.map((s) => (
                <div key={s.scenario}>
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-[12.5px] text-ink">{s.label}</span>
                    <span
                      className="num text-[14px]"
                      style={{
                        color: s.delta < 0 ? 'var(--color-good)' : 'var(--color-ink-dim)',
                      }}
                    >
                      {signed(s.delta)} s
                    </span>
                  </div>
                  <div className="mt-1">
                    <Beam
                      mean={s.delta}
                      sd={s.sd}
                      domain={[-2, 0.5]}
                      colour={s.delta < 0 ? 'var(--color-good)' : 'var(--color-residual)'}
                      zero
                      height={11}
                    />
                  </div>
                  <p className="mt-1 max-w-[48ch] text-[10.5px] leading-relaxed text-ink-faint">
                    {s.note}
                  </p>
                </div>
              ))}
            </div>
          </Panel>

          {!trust ? (
            <Panel title="Should you believe this?" aside="re-fitting under perturbed assumptions">
              <Loading what="the robustness check" />
            </Panel>
          ) : (
            <Panel
              title="Should you believe this?"
              aside={`${trust.applicability.risk} risk`}
            >
              <div className="mb-3 flex items-center gap-3">
                <div className="h-1.5 flex-1 bg-raised">
                  <div
                    className="h-full"
                    style={{
                      width: `${trust.applicability.applicability * 100}%`,
                      background:
                        trust.applicability.risk === 'low'
                          ? 'var(--color-good)'
                          : trust.applicability.risk === 'medium'
                            ? 'var(--color-medium)'
                            : 'var(--color-alert)',
                    }}
                  />
                </div>
                <span className="num shrink-0 text-[13px]">
                  {(trust.applicability.applicability * 100).toFixed(0)}%
                </span>
              </div>
              <ul className="space-y-1.5">
                {trust.applicability.reasons.map((reason, i) => (
                  <li
                    key={i}
                    className="border-l-2 border-line pl-2.5 text-[11.5px] leading-relaxed text-ink-dim"
                  >
                    {reason}
                  </li>
                ))}
              </ul>

              {trust.value_of_information.length > 0 && (
                <div className="mt-4 border-t border-line pt-3">
                  <div className="mb-2 text-[11px] text-ink-faint">
                    What would most reduce the uncertainty
                  </div>
                  {trust.value_of_information.slice(0, 3).map((v) => (
                    <div key={v.signal} className="mb-2">
                      <div className="flex items-baseline justify-between">
                        <span className="text-[11.5px] text-ink-dim">{v.signal}</span>
                        <span className="num text-[11.5px] text-alert">
                          −{(v.estimated_reduction * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="max-w-[46ch] text-[10.5px] leading-relaxed text-ink-faint">
                        {v.rationale}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </Panel>
          )}
        </div>
      </div>
    </div>
  )
}

function HealthChart({ timeline }: { timeline: HealthTimeline }) {
  const colour = compoundColourResolved(timeline.compound)
  const ages = timeline.rows.map((r) => r.tyre_age)

  const option = {
    animationDuration: 500,
    grid: { left: 46, right: 46, top: 26, bottom: 36 },
    legend: {
      top: 0,
      data: ['Performance lost', 'Health'],
      textStyle: { color: '#8fa3ae', fontSize: 10 },
      itemWidth: 14,
      itemHeight: 2,
    },
    xAxis: {
      type: 'category',
      data: ages,
      name: 'tyre age (laps)',
      nameLocation: 'middle',
      nameGap: 22,
      nameTextStyle: { color: '#5d6f7a', fontSize: 10.5 },
      axisLine: { lineStyle: { color: '#26343d' } },
      axisLabel: { color: '#5d6f7a', fontSize: 10.5 },
    },
    yAxis: [
      {
        type: 'value',
        name: 'seconds lost',
        nameTextStyle: { color: '#5d6f7a', fontSize: 10.5 },
        axisLine: { show: false },
        axisLabel: { color: '#5d6f7a', fontSize: 10.5, formatter: (v: number) => v.toFixed(1) },
        splitLine: { lineStyle: { color: '#1d272e' } },
      },
      {
        type: 'value',
        name: 'health',
        min: 0,
        max: 100,
        nameTextStyle: { color: '#5d6f7a', fontSize: 10.5 },
        axisLine: { show: false },
        axisLabel: { color: '#5d6f7a', fontSize: 10.5 },
        splitLine: { show: false },
      },
    ],
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#151d23',
      borderColor: '#26343d',
      textStyle: { color: '#e4eaed', fontSize: 11.5 },
      // Only the two real series. The band is drawn as a transparent base plus
      // a stacked ribbon, and surfacing those would show the reader a stack
      // offset labelled as if it were a measurement.
      formatter: (params: { axisValue: string; seriesName: string; value: number }[]) => {
        const age = params[0]?.axisValue
        const rows = params
          .filter((p) => p.seriesName === 'Performance lost' || p.seriesName === 'Health')
          .map((p) =>
            p.seriesName === 'Health'
              ? `Health ${p.value.toFixed(0)} / 100`
              : `Lost ${p.value.toFixed(2)} s/lap vs fresh`,
          )
        return `Tyre age ${age} laps<br/>${rows.join('<br/>')}`
      },
    },
    series: [
      // Uncertainty band, drawn as a transparent base plus a visible ribbon.
      {
        name: 'band base',
        type: 'line',
        stack: 'band',
        symbol: 'none',
        lineStyle: { opacity: 0 },
        data: timeline.rows.map((r) => Math.max(r.level - 1.96 * r.level_sd, 0)),
        silent: true,
        tooltip: { show: false },
      },
      {
        name: 'band width',
        type: 'line',
        stack: 'band',
        symbol: 'none',
        lineStyle: { opacity: 0 },
        areaStyle: { color: colour, opacity: 0.14 },
        data: timeline.rows.map((r) => 2 * 1.96 * r.level_sd),
        silent: true,
        tooltip: { show: false },
      },
      {
        name: 'Performance lost',
        type: 'line',
        symbol: 'none',
        smooth: 0.2,
        lineStyle: { color: colour, width: 2.4 },
        data: timeline.rows.map((r) => r.level),
        z: 5,
      },
      {
        name: 'Health',
        type: 'line',
        yAxisIndex: 1,
        symbol: 'none',
        smooth: 0.2,
        lineStyle: { color: resolveColour('var(--color-good)'), width: 1.2, type: 'dashed' },
        data: timeline.rows.map((r) => r.health),
        z: 4,
      },
    ],
  }

  return (
    <>
      <ReactECharts option={option} style={{ height: 250 }} notMerge />
      <p className="mt-2 max-w-[60ch] text-[11px] leading-relaxed text-ink-faint">
        {timeline.health_anchor_note} The shaded band is the 95% range — it widens
        where the model has less to go on.
      </p>
    </>
  )
}
