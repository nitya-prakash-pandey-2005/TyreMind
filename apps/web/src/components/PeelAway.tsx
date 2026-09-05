/**
 * The hero: confounders lifted off a stint one at a time.
 *
 * The product's whole claim is that observed pace is not tyre degradation. The
 * most direct way to make that land is to show it happening -- start from the
 * lap times a stint actually set, then remove fuel, then track, then traffic,
 * and let the tyre curve underneath be the thing left standing.
 *
 * This is the one orchestrated motion in the application. Everything else moves
 * only in response to a click.
 */

import { useEffect, useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { DecompositionRow } from '../lib/api'
import { compoundColourResolved } from '../lib/api'
import { useThemeColours } from '../lib/theme'

/** Peel order: largest confounder first, so each step visibly changes the shape. */
const STEPS = [
  { key: 'none', label: 'Lap times as driven', removes: null as string | null },
  { key: 'fuel', label: 'Fuel burn-off removed', removes: 'fuel' },
  { key: 'track', label: 'Track evolution removed', removes: 'track' },
  { key: 'traffic', label: 'Traffic removed', removes: 'traffic' },
] as const

export function PeelAway({
  rows,
  compound,
  autoPlay = true,
}: {
  rows: DecompositionRow[]
  compound: string
  autoPlay?: boolean
}) {
  const [step, setStep] = useState(0)

  useEffect(() => {
    if (!autoPlay || step >= STEPS.length - 1) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const timer = setTimeout(() => setStep((s) => s + 1), reduced ? 400 : 1600)
    return () => clearTimeout(timer)
  }, [step, autoPlay])

  const series = useMemo(() => {
    const removed = STEPS.slice(1, step + 1)
      .map((s) => s.removes)
      .filter(Boolean) as string[]

    return rows.map((row) => {
      let value = row.observed_delta
      for (const term of removed) {
        value -= row[term as 'fuel' | 'track' | 'traffic'] ?? 0
      }
      return [row.tyre_age, value]
    })
  }, [rows, step])

  const tyreOnly = useMemo(
    () => rows.map((row) => [row.tyre_age, row.tyre ?? 0]),
    [rows],
  )

  const colour = compoundColourResolved(compound)
  const atEnd = step === STEPS.length - 1
  const c = useThemeColours()

  const option = {
    // The morph between steps is the point of this chart, so update animation
    // stays on. The *entry* animation does not: ECharts draws a line by
    // expanding a clip path, and autoplay advances the step every 1.6 s. An
    // entry animation cut short by the next step leaves that clip frozen
    // part-way, which renders as a curve that stops dead in mid-air.
    animation: true,
    animationDuration: 0,
    animationDurationUpdate: 700,
    animationEasingUpdate: 'cubicInOut',
    grid: { left: 46, right: 18, top: 18, bottom: 34 },
    xAxis: {
      type: 'value',
      name: 'tyre age (laps)',
      nameLocation: 'middle',
      nameGap: 22,
      nameTextStyle: { color: c.inkFaint, fontSize: 11 },
      axisLine: { lineStyle: { color: c.line } },
      axisLabel: { color: c.inkFaint, fontSize: 11 },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: 'seconds vs stint start',
      nameLocation: 'middle',
      nameGap: 34,
      nameTextStyle: { color: c.inkFaint, fontSize: 11 },
      axisLine: { show: false },
      axisLabel: { color: c.inkFaint, fontSize: 11, formatter: (v: number) => v.toFixed(1) },
      splitLine: { lineStyle: { color: c.raised } },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: c.surface,
      borderColor: c.line,
      textStyle: { color: c.ink, fontSize: 12 },
      valueFormatter: (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)} s`,
    },
    series: [
      {
        id: 'tyre',
        name: 'True tyre degradation',
        type: 'line',
        data: tyreOnly,
        smooth: 0.25,
        symbol: 'none',
        lineStyle: { color: colour, width: 2.5, type: 'dashed' },
        z: 3,
      },
      {
        id: 'peel',
        name: STEPS[step].label,
        type: 'line',
        data: series,
        smooth: 0.2,
        symbol: 'circle',
        symbolSize: 4,
        itemStyle: { color: atEnd ? colour : c.inkDim },
        lineStyle: { color: atEnd ? colour : c.inkDim, width: 2 },
        z: 4,
      },
    ],
  }

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        {STEPS.map((s, i) => (
          <button
            key={s.key}
            onClick={() => setStep(i)}
            className={`border px-2.5 py-1 text-[11px] transition-colors ${
              i === step
                ? 'border-alert text-alert'
                : i < step
                  ? 'border-line text-ink-dim hover:border-line-bright'
                  : 'border-line text-ink-faint hover:border-line-bright'
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      <ReactECharts option={option} style={{ height: 300 }} />

      <p className="mt-3 max-w-[68ch] text-[12.5px] leading-relaxed text-ink-dim">
        {atEnd ? (
          <>
            With fuel, track evolution and traffic taken out, what is left is the tyre.
            The dashed line is the model&rsquo;s estimate of the degradation curve; the
            solid line is the measured lap times with the confounders removed. The gap
            between them is lap-to-lap driver noise the model does not try to explain --
            they track each other, they do not coincide, and a model claiming they did
            would be fitting the noise.
          </>
        ) : (
          <>
            The grey line is what the stopwatch saw. It is not the tyre &mdash; not yet.
            Each step removes one cause that changed the lap time without the tyre having
            changed at all.
          </>
        )}
      </p>
    </div>
  )
}
