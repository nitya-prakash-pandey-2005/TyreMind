/**
 * The chart set.
 *
 * Every chart here answers a question the surrounding numbers cannot. None of
 * them is decoration, and where one would only restate a table it is not built.
 *
 * All of them share `useAxis`, which reads concrete colours from the theme —
 * ECharts draws to canvas and cannot resolve CSS custom properties, so a chart
 * that hard-codes hex values is invisible in the other theme.
 */

import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { useThemeColours } from '../lib/theme'
import { compoundColour } from '../lib/api'

/** Axis and tooltip styling shared by every chart, so they read as one system. */
function useAxis() {
  const c = useThemeColours()

  return useMemo(() => {
    const axisLabel = { color: c.inkFaint, fontSize: 10 }
    const nameTextStyle = { color: c.inkFaint, fontSize: 10 }
    return {
      colours: c,
      base: {
        // No entry animation. These are analytical charts, not a title
        // sequence, and every one of them is re-initialised with `notMerge`
        // when the session or theme changes -- an interrupted entry animation
        // leaves the line's expanding clip path frozen part-way, which renders
        // as a curve that silently stops early. That is indistinguishable from
        // a data bug, so the animation is not worth its risk here.
        animation: false,
        tooltip: {
          backgroundColor: c.surface,
          borderColor: c.line,
          textStyle: { color: c.ink, fontSize: 11.5 },
        },
        legend: {
          top: 0,
          textStyle: { color: c.inkDim, fontSize: 10.5 },
          itemWidth: 14,
          itemHeight: 2,
        },
      },
      xAxis: (name: string) => ({
        name,
        nameLocation: 'middle' as const,
        nameGap: 22,
        nameTextStyle,
        axisLabel,
        axisLine: { lineStyle: { color: c.line } },
        splitLine: { show: false },
      }),
      yAxis: (name: string) => ({
        name,
        nameTextStyle,
        axisLabel,
        axisLine: { show: false },
        splitLine: { lineStyle: { color: c.raised } },
      }),
    }
  }, [c])
}

// --------------------------------------------------------------------------
// Overview
// --------------------------------------------------------------------------

export interface CompoundCurve {
  compound: string
  rate: number
  sd: number
  maxAge: number
}

/**
 * The fitted degradation curve per compound, with its uncertainty fanning out.
 *
 * Plots the model's own estimate -- rate x age, with the band from the rate's
 * posterior -- rather than an average of per-car latent states at each age.
 *
 * That distinction matters and the first version got it wrong. Cars start runs
 * at different tyre ages, because a scrubbed set arrives with laps already on
 * it. So at "age 5" one car is five laps into a stint and another has just
 * fitted a set, and averaging their accumulated loss produces a number that
 * describes neither. Requiring three observations per age then truncated the
 * curve wherever the field thinned out, which was almost immediately.
 *
 * The band widening with age is the honest part: uncertainty in a *rate*
 * compounds the further you extrapolate it.
 */
export function DegradationCurves({ curves }: { curves: CompoundCurve[] }) {
  const { base, xAxis, yAxis } = useAxis()

  const option = useMemo(() => {
    const horizon = Math.max(...curves.map((c) => c.maxAge), 10)
    const ages = Array.from({ length: Math.ceil(horizon) + 1 }, (_, i) => i)

    const series = curves.flatMap((c) => {
      const colour = compoundColour(c.compound)
      // Only draw as far as this compound was actually run. Extending every
      // curve to the longest stint in the session would show extrapolation as
      // if it were estimate.
      const own = ages.filter((a) => a <= c.maxAge)
      const mean = own.map((a) => [a, c.rate * a])
      const halfBand = own.map((a) => 1.96 * c.sd * a)

      return [
        {
          name: `${c.compound} base`,
          type: 'line',
          stack: `band-${c.compound}`,
          symbol: 'none',
          lineStyle: { opacity: 0 },
          data: own.map((a, i) => [a, c.rate * a - halfBand[i]]),
          silent: true,
          tooltip: { show: false },
          legendHoverLink: false,
        },
        {
          name: `${c.compound} band`,
          type: 'line',
          stack: `band-${c.compound}`,
          symbol: 'none',
          lineStyle: { opacity: 0 },
          areaStyle: { color: colour, opacity: 0.13 },
          data: own.map((a, i) => [a, 2 * halfBand[i]]),
          silent: true,
          tooltip: { show: false },
          legendHoverLink: false,
        },
        {
          name: c.compound,
          type: 'line',
          data: mean,
          symbol: 'none',
          smooth: false,
          lineStyle: { color: colour, width: 2.6 },
          z: 5,
        },
      ]
    })

    return {
      ...base,
      legend: { ...base.legend, data: curves.map((c) => c.compound) },
      grid: { left: 54, right: 20, top: 28, bottom: 38 },
      xAxis: { type: 'value', min: 0, ...xAxis('tyre age (laps)') },
      yAxis: { type: 'value', ...yAxis('seconds lost vs fresh') },
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        valueFormatter: (v: number) => (v == null ? '—' : `${v.toFixed(2)} s`),
      },
      series,
    }
  }, [curves, base, xAxis, yAxis])

  return <ReactECharts option={option} style={{ height: 260 }} notMerge />
}

/** How many laps survived filtering, and what removed the rest. */
export function QualityBreakdown({
  retained,
  exclusions,
  labels,
}: {
  retained: number
  exclusions: Record<string, number>
  labels: Record<string, string>
}) {
  const { base, colours } = useAxis()

  const option = useMemo(() => {
    const entries = Object.entries(exclusions).sort((a, b) => b[1] - a[1])
    const data = [
      { name: 'Analysed', value: retained, itemStyle: { color: colours.alert } },
      ...entries.map(([reason, count], i) => ({
        name: labels[reason] ?? reason,
        value: count,
        itemStyle: {
          // Excluded reasons share one cool hue at decreasing opacity, so the
          // retained slice is the only thing that reads as a result.
          color: colours.residual,
          opacity: 0.85 - i * 0.11,
        },
      })),
    ]

    return {
      ...base,
      legend: { show: false },
      tooltip: {
        ...base.tooltip,
        formatter: (p: { name: string; value: number; percent: number }) =>
          `${p.name}<br/>${p.value} laps (${p.percent.toFixed(0)}%)`,
      },
      series: [
        {
          type: 'pie',
          radius: ['52%', '78%'],
          center: ['50%', '52%'],
          itemStyle: { borderColor: colours.surface, borderWidth: 2 },
          label: {
            show: true,
            color: colours.inkFaint,
            fontSize: 10,
            formatter: '{b}\n{c}',
          },
          labelLine: { lineStyle: { color: colours.line } },
          data,
        },
      ],
    }
  }, [retained, exclusions, labels, base, colours])

  return <ReactECharts option={option} style={{ height: 250 }} notMerge />
}

// --------------------------------------------------------------------------
// Explain
// --------------------------------------------------------------------------

export interface DecompositionRow {
  tyre_age: number
  tyre?: number
  fuel?: number
  track?: number
  traffic?: number
  residual: number
  observed_delta: number
}

/**
 * Where the lap time went, lap by lap across a stint.
 *
 * The waterfall shows one lap. This shows the whole stint, so you can watch the
 * tyre contribution grow while the fuel contribution falls — and see the exact
 * lap where the tyre overtakes fuel, which is the moment a stint turns.
 *
 * Positive and negative terms are stacked separately, because a single stack
 * containing both would cancel and hide the size of each.
 */
export function StintDecomposition({ rows }: { rows: DecompositionRow[] }) {
  const { base, xAxis, yAxis, colours } = useAxis()

  const option = useMemo(() => {
    const ages = rows.map((r) => r.tyre_age)
    const terms: { key: keyof DecompositionRow; label: string; colour: string }[] = [
      { key: 'tyre', label: 'Tyre', colour: colours.alert },
      { key: 'fuel', label: 'Fuel burn-off', colour: colours.fuel },
      { key: 'track', label: 'Track evolution', colour: colours.track },
      { key: 'traffic', label: 'Traffic', colour: colours.traffic },
      { key: 'residual', label: 'Unexplained', colour: colours.residual },
    ]

    return {
      ...base,
      legend: { ...base.legend, data: [...terms.map((t) => t.label), 'Observed'] },
      grid: { left: 52, right: 18, top: 28, bottom: 36 },
      xAxis: { type: 'category', data: ages, ...xAxis('tyre age (laps)') },
      yAxis: { type: 'value', ...yAxis('seconds vs stint start') },
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        valueFormatter: (v: number) =>
          v == null ? '—' : `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)} s`,
      },
      series: [
        ...terms.map((t) => ({
          name: t.label,
          type: 'bar',
          stack: 'terms',
          data: rows.map((r) => (r[t.key] as number) ?? 0),
          itemStyle: { color: t.colour, opacity: t.key === 'tyre' ? 1 : 0.8 },
          barCategoryGap: '18%',
        })),
        {
          name: 'Observed',
          type: 'line',
          data: rows.map((r) => r.observed_delta),
          symbol: 'none',
          lineStyle: { color: colours.ink, width: 2, type: 'dashed' },
          z: 8,
        },
      ],
    }
  }, [rows, base, xAxis, yAxis, colours])

  return <ReactECharts option={option} style={{ height: 280 }} notMerge />
}

/** The circuit gaining grip over the session, with its uncertainty. */
export function TrackEvolutionChart({
  rows,
}: {
  rows: { session_lap: number; track_effect: number; track_effect_sd: number }[]
}) {
  const { base, xAxis, yAxis, colours } = useAxis()

  const option = useMemo(
    () => ({
      ...base,
      legend: { show: false },
      grid: { left: 52, right: 18, top: 18, bottom: 36 },
      xAxis: {
        type: 'category',
        data: rows.map((r) => r.session_lap),
        ...xAxis('session lap'),
        // A label per lap is unreadable over a 53-lap race and carries nothing:
        // the shape is the message, not any individual lap number.
        axisLabel: { ...xAxis('session lap').axisLabel, interval: (i: number) => i % 5 === 0 },
      },
      yAxis: { type: 'value', ...yAxis('seconds the circuit has gained') },
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        valueFormatter: (v: number) => (v == null ? '—' : `${v.toFixed(2)} s`),
      },
      series: [
        {
          name: 'base',
          type: 'line',
          stack: 'band',
          symbol: 'none',
          lineStyle: { opacity: 0 },
          data: rows.map((r) => -r.track_effect - 1.96 * r.track_effect_sd),
          silent: true,
          tooltip: { show: false },
        },
        {
          name: 'band',
          type: 'line',
          stack: 'band',
          symbol: 'none',
          lineStyle: { opacity: 0 },
          areaStyle: { color: colours.track, opacity: 0.15 },
          data: rows.map((r) => 2 * 1.96 * r.track_effect_sd),
          silent: true,
          tooltip: { show: false },
        },
        {
          name: 'Track effect',
          type: 'line',
          data: rows.map((r) => -r.track_effect),
          symbol: 'none',
          smooth: 0.3,
          lineStyle: { color: colours.track, width: 2.2 },
          z: 5,
        },
      ],
    }),
    [rows, base, xAxis, yAxis, colours],
  )

  return <ReactECharts option={option} style={{ height: 210 }} notMerge />
}

// --------------------------------------------------------------------------
// Strategy
// --------------------------------------------------------------------------

export interface PitSweepRow {
  pit_lap: number
  expected_time: number
  downside: number
  best_case: number
  runs_past_cliff: number
}

/**
 * Expected race time for every possible pit lap.
 *
 * The most useful strategy picture there is. A table of four options says which
 * is best; the sweep says how *sharp* the optimum is — whether stopping two laps
 * late costs a tenth or costs the race. A flat curve is itself the answer:
 * the decision does not matter much, so spend the attention elsewhere.
 */
export function PitWindowChart({
  sweep,
  optimum,
  window,
  stayOut,
}: {
  sweep: PitSweepRow[]
  optimum: number
  window: [number, number] | null
  stayOut: number
}) {
  const { base, xAxis, yAxis, colours } = useAxis()

  const option = useMemo(() => {
    const bestStop = Math.min(...sweep.map((r) => r.expected_time))
    // Baseline on the best action available, which is not always a stop. When
    // staying out wins, baselining on the best pit lap puts the stay-out
    // reference line below zero and off the axis -- hiding the one fact the
    // screen exists to deliver. Absolute race time is a four-digit number whose
    // interesting variation is in the last two, so something has to be zero.
    const baseline = Math.min(bestStop, stayOut)
    const relative = sweep.map((r) => [r.pit_lap, r.expected_time - baseline])
    const downside = sweep.map((r) => [r.pit_lap, r.downside - baseline])

    return {
      ...base,
      legend: { ...base.legend, data: ['Expected', 'Bad case'] },
      grid: { left: 52, right: 18, top: 28, bottom: 36 },
      // Only the candidate laps exist -- letting the axis run from zero would
      // spend half the width on laps the car has already completed.
      xAxis: {
        type: 'value',
        min: sweep[0].pit_lap - 1,
        max: sweep[sweep.length - 1].pit_lap + 1,
        ...xAxis('pit on lap'),
      },
      yAxis: { type: 'value', ...yAxis('seconds vs the best option') },
      tooltip: {
        ...base.tooltip,
        trigger: 'axis',
        valueFormatter: (v: number) => (v == null ? '—' : `+${v.toFixed(2)} s`),
      },
      series: [
        {
          name: 'Bad case',
          type: 'line',
          data: downside,
          symbol: 'none',
          smooth: 0.2,
          lineStyle: { color: colours.residual, width: 1.2, type: 'dashed' },
        },
        {
          name: 'Expected',
          type: 'line',
          data: relative,
          symbol: 'none',
          smooth: 0.2,
          lineStyle: { color: colours.alert, width: 2.6 },
          areaStyle: { color: colours.alert, opacity: 0.1 },
          markPoint: {
            symbol: 'circle',
            symbolSize: 9,
            itemStyle: { color: colours.good },
            label: { show: false },
            data: [{ xAxis: optimum, yAxis: bestStop - baseline }],
          },
          markArea: window
            ? {
                itemStyle: { color: colours.good, opacity: 0.08 },
                data: [[{ xAxis: window[0] }, { xAxis: window[1] }]],
              }
            : undefined,
          markLine: {
            symbol: 'none',
            label: {
              color: colours.inkFaint,
              fontSize: 10,
              formatter: 'stay out — no stop at all',
              position: 'insideEndTop',
            },
            lineStyle: { color: colours.inkFaint, type: 'dotted' },
            data: [{ yAxis: stayOut - baseline }],
          },
          z: 5,
        },
      ],
    }
  }, [sweep, optimum, window, stayOut, base, xAxis, yAxis, colours])

  return <ReactECharts option={option} style={{ height: 260 }} notMerge />
}

// --------------------------------------------------------------------------
// Beyond racing
// --------------------------------------------------------------------------

/**
 * Predicted against actual remaining life, on NASA's turbofan benchmark.
 *
 * The honest way to show a prognostics result. Points on the diagonal are exact;
 * points *below* it predicted less life than the engine had, which is the safe
 * direction to be wrong. The asymmetry is the whole point of the NASA scoring
 * function, and a scatter shows it where an RMSE cannot.
 */
export function RulScatter({
  predictions,
  truths,
}: {
  predictions: number[]
  truths: number[]
}) {
  const { base, xAxis, yAxis, colours } = useAxis()

  const option = useMemo(() => {
    const points = truths.map((t, i) => [t, predictions[i]])
    const limit = Math.ceil(Math.max(...truths, ...predictions) / 25) * 25

    return {
      ...base,
      legend: { show: false },
      grid: { left: 52, right: 20, top: 20, bottom: 36 },
      xAxis: { type: 'value', min: 0, max: limit, ...xAxis('actual cycles remaining') },
      yAxis: { type: 'value', min: 0, max: limit, ...yAxis('predicted') },
      tooltip: {
        ...base.tooltip,
        formatter: (p: { value: [number, number] }) =>
          `Predicted ${p.value[1].toFixed(0)} cycles<br/>Actual ${p.value[0].toFixed(0)}<br/>` +
          `${p.value[1] < p.value[0] ? 'early — the safe direction' : 'late'}`,
      },
      series: [
        {
          name: 'perfect',
          type: 'line',
          data: [
            [0, 0],
            [limit, limit],
          ],
          symbol: 'none',
          lineStyle: { color: colours.inkFaint, width: 1, type: 'dashed' },
          silent: true,
        },
        {
          type: 'scatter',
          data: points,
          symbolSize: 7,
          itemStyle: {
            color: (p: { value: [number, number] }) =>
              p.value[1] < p.value[0] ? colours.good : colours.alert,
            opacity: 0.75,
          },
        },
      ],
    }
  }, [predictions, truths, base, xAxis, yAxis, colours])

  return <ReactECharts option={option} style={{ height: 280 }} notMerge />
}

// --------------------------------------------------------------------------
// Circuit
// --------------------------------------------------------------------------

/**
 * The hardest-working sections of a lap, ranked.
 *
 * The 3D line shows where the load is; this says how much, and how concentrated.
 * On a power circuit a handful of braking zones account for most of the damage;
 * on a flowing one it is spread out. That difference is why the same compound
 * behaves so differently between them.
 */
export function LoadHotspots({
  tyreLoad,
  speed,
  topN = 10,
}: {
  tyreLoad: number[]
  speed: number[]
  topN?: number
}) {
  const { base, xAxis, yAxis, colours } = useAxis()

  const option = useMemo(() => {
    // Group contiguous high-load samples into "sections", so the ranking lists
    // corners rather than individual telemetry points.
    const threshold = 0.45
    const sections: { start: number; end: number; peak: number; energy: number }[] = []
    let current: { start: number; end: number; peak: number; energy: number } | null = null

    tyreLoad.forEach((v, i) => {
      if (v >= threshold) {
        if (!current) current = { start: i, end: i, peak: v, energy: 0 }
        current.end = i
        current.peak = Math.max(current.peak, v)
        current.energy += v
      } else if (current) {
        sections.push(current)
        current = null
      }
    })
    if (current) sections.push(current)

    const ranked = sections.sort((a, b) => b.energy - a.energy).slice(0, topN).reverse()
    const total = tyreLoad.reduce((a, b) => a + b, 0) || 1

    return {
      ...base,
      legend: { show: false },
      grid: { left: 92, right: 26, top: 14, bottom: 36 },
      xAxis: { type: 'value', ...xAxis('share of the lap total (%)') },
      yAxis: {
        type: 'category',
        data: ranked.map(
          (s) =>
            `${Math.round((s.start / tyreLoad.length) * 100)}% · ${Math.round(
              speed[Math.round((s.start + s.end) / 2)] ?? 0,
            )} km/h`,
        ),
        axisLabel: { color: colours.inkFaint, fontSize: 9.5 },
        axisLine: { show: false },
        splitLine: { show: false },
      },
      tooltip: {
        ...base.tooltip,
        formatter: (p: { name: string; value: number }) =>
          `${p.name} into the lap<br/>${p.value.toFixed(1)}% of total tyre loading`,
      },
      series: [
        {
          type: 'bar',
          data: ranked.map((s) => (100 * s.energy) / total),
          itemStyle: { color: colours.alert, opacity: 0.85 },
          barWidth: '62%',
        },
      ],
    }
  }, [tyreLoad, speed, topN, base, xAxis, yAxis, colours])

  return <ReactECharts option={option} style={{ height: 250 }} notMerge />
}
