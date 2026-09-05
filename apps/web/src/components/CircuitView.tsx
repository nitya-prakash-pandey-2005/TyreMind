/**
 * Where on the lap the tyre gets used up.
 *
 * The 3D line answers a question the degradation number cannot: a rate of
 * 0.11 s/lap tells you the tyre is going away, but not that three quarters of
 * the damage happens in four corners. That is actionable — it is what a driver
 * can change.
 *
 * Alongside it, the g-g diagram. It is the standard way vehicle dynamicists
 * read a lap: every point is one instant, plotted by how much grip was being
 * spent turning versus stopping. A car exploiting its tyres fills the circle;
 * one that does not leaves the edges empty.
 */

import { useEffect, useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import {
  Circuit3D,
  CircuitLegend,
  MODE_HELP,
  MODE_LABEL,
  useLapProgress,
  type ColourMode,
  type TrackGeometry,
} from './Circuit3D'
import { ErrorNote, Loading, Panel, Stat } from './primitives'
import { Explainer } from './Explainer'
import { useThemeColours } from '../lib/theme'

const MODES: ColourMode[] = ['tyre_load', 'speed_kmh', 'lateral_g']

export function CircuitView({ circuit }: { circuit: string }) {
  const [track, setTrack] = useState<TrackGeometry | null>(null)
  const [error, setError] = useState('')
  const [mode, setMode] = useState<ColourMode>('tyre_load')
  const [rotating, setRotating] = useState(true)
  const progress = useLapProgress(rotating)

  useEffect(() => {
    setTrack(null)
    setError('')
    fetch(`/api/physics/track-geometry?circuit=${encodeURIComponent(circuit)}`)
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText)
        return r.json()
      })
      .then(setTrack)
      .catch((e) => setError(String(e.message ?? e)))
  }, [circuit])

  if (error) {
    return (
      <div className="space-y-3">
        <Explainer id="circuit" question="What is this view for?">
          <p>
            It shows <strong>where on the lap</strong> the tyres get used up, using
            the real racing line recorded from the car.
          </p>
        </Explainer>
        <ErrorNote error={error} />
      </div>
    )
  }
  if (!track) return <Loading what="the circuit geometry" />

  const stats = track.stats

  return (
    <div className="space-y-3">
      <Explainer id="circuit" question="What am I looking at?">
        <p>
          This is the <strong>actual racing line</strong> a car drove, recorded by
          GPS at up to ten times a second, drawn in three dimensions with the
          elevation exaggerated so hills are visible.
        </p>
        <p>
          The colour is the interesting part. It is not speed or height — it is how
          hard the tyres are working at that point, computed from how tightly the
          car is turning and how hard it is braking.{' '}
          <strong>
            Bright means the rubber is being consumed.
          </strong>{' '}
          A degradation rate tells you a tyre is going away; this tells you where.
        </p>
      </Explainer>

      <div className="grid gap-3 xl:grid-cols-[1.4fr_1fr]">
        <Panel
          title={`${track.circuit} — fastest lap, ${track.year}`}
          aside={`${stats.driver} · ${stats.lap_time_s ? stats.lap_time_s.toFixed(3) + 's' : '—'}`}
        >
          <div className="mb-3 flex flex-wrap items-center gap-2">
            {MODES.map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`border px-2.5 py-1 text-[11px] transition-colors ${
                  m === mode
                    ? 'border-alert text-alert'
                    : 'border-line text-ink-dim hover:border-line-bright'
                }`}
              >
                {MODE_LABEL[m]}
              </button>
            ))}
            <button
              onClick={() => setRotating((r) => !r)}
              className="ml-auto border border-line px-2.5 py-1 text-[11px] text-ink-dim transition-colors hover:border-line-bright"
            >
              {rotating ? 'Pause' : 'Rotate'}
            </button>
          </div>

          <div className="h-[400px] w-full border border-line">
            <Circuit3D track={track} mode={mode} rotating={rotating} progress={progress} />
          </div>

          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <CircuitLegend mode={mode} />
            <span className="text-[10.5px] text-ink-faint">
              {stats.n_points} telemetry points
            </span>
          </div>

          <p className="mt-2 max-w-[74ch] text-[11.5px] leading-relaxed text-ink-faint">
            {MODE_HELP[mode]}
          </p>
        </Panel>

        <div className="space-y-3">
          <Panel title="This lap">
            <div className="grid grid-cols-2 gap-5">
              <Stat
                label="Track length"
                value={(stats.track_length_m / 1000).toFixed(2)}
                unit="km"
              />
              <Stat label="Top speed" value={stats.top_speed_kmh.toFixed(0)} unit="km/h" />
              <Stat
                label="Peak cornering"
                value={stats.peak_lateral_g.toFixed(1)}
                unit="g"
                tone="warm"
                hint="99th percentile, not the max"
              />
              <Stat
                label="Peak braking"
                value={stats.peak_braking_g.toFixed(1)}
                unit="g"
              />
            </div>
            <p className="mt-4 max-w-[46ch] text-[11px] leading-relaxed text-ink-faint">
              Peak loads are reported as the 99th percentile rather than the maximum.
              A single GPS glitch produces one impossible sample, and quoting the
              maximum would quote the glitch.
            </p>
          </Panel>

          <Panel title="The friction circle" aside="every point is one instant">
            <GgDiagram track={track} />
          </Panel>
        </div>
      </div>

      <Panel title="Speed and load around the lap">
        <TraceChart track={track} />
      </Panel>
    </div>
  )
}

/**
 * The g-g diagram: lateral against longitudinal acceleration.
 *
 * Reads as a circle because a tyre has one friction budget and can spend it on
 * turning, on stopping, or on a combination — but not on more than the total.
 * The filled shape is the envelope the car actually used.
 */
function GgDiagram({ track }: { track: TrackGeometry }) {
  const colours = useThemeColours()

  const option = useMemo(() => {
    const points = track.lateral_g.map((lat, i) => [lat, track.longitudinal_g[i]])
    const limit =
      Math.ceil(
        Math.max(
          ...track.lateral_g.map(Math.abs),
          ...track.longitudinal_g.map(Math.abs),
        ),
      ) + 0.5

    return {
      animation: false,
      grid: { left: 42, right: 16, top: 16, bottom: 34 },
      xAxis: {
        type: 'value',
        min: -limit,
        max: limit,
        name: 'cornering (g)',
        nameLocation: 'middle',
        nameGap: 21,
        nameTextStyle: { color: colours.inkFaint, fontSize: 10 },
        axisLine: { lineStyle: { color: colours.line } },
        axisLabel: { color: colours.inkFaint, fontSize: 10 },
        splitLine: { lineStyle: { color: colours.raised } },
      },
      yAxis: {
        type: 'value',
        min: -limit,
        max: limit,
        name: 'braking / accel (g)',
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: { color: colours.inkFaint, fontSize: 10 },
        axisLine: { lineStyle: { color: colours.line } },
        axisLabel: { color: colours.inkFaint, fontSize: 10 },
        splitLine: { lineStyle: { color: colours.raised } },
      },
      tooltip: {
        backgroundColor: colours.surface,
        borderColor: colours.line,
        textStyle: { color: colours.ink, fontSize: 11 },
        formatter: (p: { value: [number, number] }) =>
          `${p.value[0].toFixed(1)} g cornering<br/>${p.value[1].toFixed(1)} g braking`,
      },
      series: [
        {
          type: 'scatter',
          data: points,
          symbolSize: 4,
          itemStyle: { color: colours.alert, opacity: 0.55 },
        },
      ],
    }
  }, [track, colours])

  return (
    <>
      <ReactECharts option={option} style={{ height: 240 }} notMerge />
      <p className="mt-2 max-w-[46ch] text-[11px] leading-relaxed text-ink-faint">
        A tyre has one grip budget, spent on turning, on stopping, or shared
        between them — never more than the total. That is why the cloud forms a
        circle. Points far from the centre are laps where the car was using
        everything the tyre had.
      </p>
    </>
  )
}

/** Speed and tyre load against distance round the lap. */
function TraceChart({ track }: { track: TrackGeometry }) {
  const colours = useThemeColours()

  const option = useMemo(() => {
    const index = track.speed_kmh.map((_, i) => i)
    return {
      animation: false,
      grid: { left: 48, right: 48, top: 28, bottom: 34 },
      legend: {
        top: 0,
        textStyle: { color: colours.inkDim, fontSize: 10.5 },
        itemWidth: 14,
        itemHeight: 2,
      },
      xAxis: {
        type: 'category',
        data: index,
        name: 'position around the lap',
        nameLocation: 'middle',
        nameGap: 21,
        nameTextStyle: { color: colours.inkFaint, fontSize: 10 },
        axisLine: { lineStyle: { color: colours.line } },
        axisLabel: { show: false },
      },
      yAxis: [
        {
          type: 'value',
          name: 'km/h',
          nameTextStyle: { color: colours.inkFaint, fontSize: 10 },
          axisLine: { show: false },
          axisLabel: { color: colours.inkFaint, fontSize: 10 },
          splitLine: { lineStyle: { color: colours.raised } },
        },
        {
          type: 'value',
          name: 'tyre load',
          min: 0,
          max: 1,
          nameTextStyle: { color: colours.inkFaint, fontSize: 10 },
          axisLine: { show: false },
          axisLabel: { color: colours.inkFaint, fontSize: 10 },
          splitLine: { show: false },
        },
      ],
      tooltip: {
        trigger: 'axis',
        backgroundColor: colours.surface,
        borderColor: colours.line,
        textStyle: { color: colours.ink, fontSize: 11 },
      },
      series: [
        {
          name: 'Speed',
          type: 'line',
          data: track.speed_kmh,
          symbol: 'none',
          smooth: 0.2,
          lineStyle: { color: colours.fuel, width: 1.5 },
        },
        {
          name: 'Tyre load',
          type: 'line',
          yAxisIndex: 1,
          data: track.tyre_load,
          symbol: 'none',
          smooth: 0.2,
          lineStyle: { color: colours.alert, width: 1.5 },
          areaStyle: { color: colours.alert, opacity: 0.12 },
        },
      ],
    }
  }, [track, colours])

  return (
    <>
      <ReactECharts option={option} style={{ height: 230 }} notMerge />
      <p className="mt-2 max-w-[80ch] text-[11.5px] leading-relaxed text-ink-faint">
        Tyre load peaks where speed dips — the corners. On a power circuit like
        Monza the load spikes are few and sharp; on a flowing circuit they are
        broad and sustained, which is why the same compound behaves so differently
        between the two.
      </p>
    </>
  )
}
