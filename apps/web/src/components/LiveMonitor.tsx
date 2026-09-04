/**
 * Live session monitor.
 *
 * Streams a session through the online estimator over a WebSocket, one lap at a
 * time in order, with no access to the future. The per-update timing is shown
 * because the real-time claim should be visible rather than captioned -- if an
 * update took 40 ms, the number on screen would say so.
 *
 * Everything here is the *filtered* estimate. The retrospective view is a
 * separate tab and is labelled differently.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { LiveState } from '../lib/api'
import { compoundColour, fixed, signed } from '../lib/api'
import { Beam, CompoundChip, Panel, Stat } from './primitives'

interface Frame {
  type: string
  index?: number
  observation?: {
    driver: string
    session_lap: number
    lap_time: number
    compound: string
    tyre_age: number
    traffic_index: number
  }
  state?: LiveState
  compound_rates?: Record<string, { mean: number; sd: number }>
  total_laps?: number
  performance?: {
    total_laps_processed: number
    n_states: number
    mean_update_ms: number
    p95_update_ms: number
    max_update_ms: number
  }
  detail?: string
}

type Status = 'idle' | 'connecting' | 'streaming' | 'complete' | 'error'

export function LiveMonitor({ sessionId }: { sessionId: string }) {
  const [status, setStatus] = useState<Status>('idle')
  const [states, setStates] = useState<Record<string, LiveState>>({})
  const [rates, setRates] = useState<Record<string, { mean: number; sd: number }>>({})
  const [progress, setProgress] = useState({ done: 0, total: 0 })
  const [feed, setFeed] = useState<Frame[]>([])
  const [perf, setPerf] = useState<Frame['performance']>()
  const [error, setError] = useState('')
  const [speed, setSpeed] = useState(0.05)
  const socketRef = useRef<WebSocket | null>(null)

  const stop = useCallback(() => {
    socketRef.current?.close()
    socketRef.current = null
  }, [])

  useEffect(() => stop, [stop])
  useEffect(() => {
    stop()
    setStatus('idle')
    setStates({})
    setRates({})
    setFeed([])
    setPerf(undefined)
    setProgress({ done: 0, total: 0 })
  }, [sessionId, stop])

  const start = useCallback(() => {
    stop()
    setStatus('connecting')
    setStates({})
    setFeed([])
    setPerf(undefined)
    setError('')

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const socket = new WebSocket(
      `${protocol}//${location.host}/ws/replay/${sessionId}?speed=${speed}`,
    )
    socketRef.current = socket

    socket.onmessage = (event) => {
      const frame: Frame = JSON.parse(event.data)
      if (frame.type === 'start') {
        setStatus('streaming')
        setProgress({ done: 0, total: frame.total_laps ?? 0 })
      } else if (frame.type === 'lap' && frame.state) {
        setStates((prev) => ({ ...prev, [frame.state!.driver]: frame.state! }))
        if (frame.compound_rates) setRates(frame.compound_rates)
        setProgress((p) => ({ ...p, done: (frame.index ?? 0) + 1 }))
        setFeed((prev) => [frame, ...prev].slice(0, 9))
      } else if (frame.type === 'complete') {
        setStatus('complete')
        setPerf(frame.performance)
      } else if (frame.type === 'error') {
        setStatus('error')
        setError(frame.detail ?? 'stream failed')
      }
    }
    socket.onerror = () => {
      setStatus('error')
      setError('WebSocket connection failed. Is the API running?')
    }
    socket.onclose = () => setStatus((s) => (s === 'streaming' ? 'complete' : s))
  }, [sessionId, speed, stop])

  const ordered = Object.values(states).sort(
    (a, b) => b.degradation_rate - a.degradation_rate,
  )
  const pct = progress.total ? (progress.done / progress.total) * 100 : 0

  return (
    <div className="space-y-3">
      <Panel
        title="Live session monitor"
        aside={
          status === 'streaming'
            ? `lap ${progress.done} of ${progress.total}`
            : status === 'complete'
              ? 'session complete'
              : 'not running'
        }
      >
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <button
            onClick={status === 'streaming' ? stop : start}
            className="border border-alert px-3.5 py-1.5 text-[12px] font-medium text-alert transition-colors hover:bg-alert/10"
          >
            {status === 'streaming' ? 'Stop stream' : 'Start live replay'}
          </button>

          <label className="flex items-center gap-2 text-[11px] text-ink-faint">
            pace
            <input
              type="range"
              min={0}
              max={0.3}
              step={0.01}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              disabled={status === 'streaming'}
              className="w-28 accent-[var(--color-alert)]"
            />
            <span className="num">{speed === 0 ? 'max' : `${(speed * 1000).toFixed(0)} ms`}</span>
          </label>

          <span
            className="flex items-center gap-1.5 text-[11px]"
            style={{ color: status === 'streaming' ? 'var(--color-good)' : 'var(--color-ink-faint)' }}
          >
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full ${
                status === 'streaming' ? 'animate-pulse' : ''
              }`}
              style={{
                background:
                  status === 'streaming' ? 'var(--color-good)' : 'var(--color-ink-faint)',
              }}
            />
            {status}
          </span>
        </div>

        <div className="h-0.5 w-full bg-raised">
          <div
            className="h-full bg-alert transition-[width] duration-200"
            style={{ width: `${pct}%` }}
          />
        </div>

        {error && <div className="mt-3 text-[12px] text-alert">{error}</div>}

        {perf && (
          <div className="mt-4 grid grid-cols-2 gap-5 border-t border-line pt-4 sm:grid-cols-4">
            <Stat label="Laps processed" value={String(perf.total_laps_processed)} />
            <Stat label="Model states" value={String(perf.n_states)} />
            <Stat
              label="Mean update"
              value={perf.mean_update_ms.toFixed(2)}
              unit="ms"
              tone="warm"
            />
            <Stat label="Worst update" value={perf.max_update_ms.toFixed(2)} unit="ms" />
          </div>
        )}
      </Panel>

      <div className="grid gap-3 lg:grid-cols-[1.35fr_1fr]">
        <Panel title="Tyre state, as known right now" aside="filtered — no future data used">
          {ordered.length === 0 ? (
            <div className="py-8 text-center text-[12px] text-ink-faint">
              Start the replay to watch the estimator build its picture lap by lap.
            </div>
          ) : (
            <div className="space-y-1">
              <div className="grid grid-cols-[54px_74px_46px_1fr_78px] gap-2 pb-1.5 text-[10px] text-ink-faint">
                <span>car</span>
                <span>compound</span>
                <span className="text-right">age</span>
                <span>degradation rate (s/lap)</span>
                <span className="text-right">health</span>
              </div>
              {ordered.map((state) => (
                <div
                  key={state.driver}
                  className="grid grid-cols-[54px_74px_46px_1fr_78px] items-center gap-2 border-t border-line/60 py-1.5"
                >
                  <span className="num text-[12.5px] font-medium">{state.driver}</span>
                  <CompoundChip compound={state.compound} />
                  <span className="num text-right text-[12px] text-ink-dim">
                    {state.tyre_age.toFixed(0)}
                  </span>
                  <div className="flex items-center gap-2">
                    <Beam
                      mean={state.degradation_rate}
                      sd={state.degradation_rate_sd}
                      domain={[-0.05, 0.3]}
                      colour={compoundColour(state.compound)}
                      zero
                      height={14}
                    />
                    <span className="num w-14 shrink-0 text-right text-[12px]">
                      {fixed(state.degradation_rate)}
                    </span>
                  </div>
                  <span
                    className="num text-right text-[12.5px]"
                    style={{
                      color:
                        state.health_index > 60
                          ? 'var(--color-good)'
                          : state.health_index > 25
                            ? 'var(--color-medium)'
                            : 'var(--color-alert)',
                    }}
                  >
                    {state.health_index.toFixed(0)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Panel>

        <div className="space-y-3">
          <Panel title="Compound baselines" aside="pooled across the field">
            {Object.keys(rates).length === 0 ? (
              <div className="py-4 text-[12px] text-ink-faint">Waiting for laps.</div>
            ) : (
              <div className="space-y-3">
                {Object.entries(rates).map(([compound, r]) => (
                  <div key={compound}>
                    <div className="mb-1 flex items-baseline justify-between">
                      <CompoundChip compound={compound} />
                      <span className="num text-[12.5px]">
                        {fixed(r.mean)}
                        <span className="ml-1 text-[10px] text-ink-faint">
                          ± {r.sd.toFixed(3)}
                        </span>
                      </span>
                    </div>
                    <Beam
                      mean={r.mean}
                      sd={r.sd}
                      domain={[-0.05, 0.25]}
                      colour={compoundColour(compound)}
                      zero
                      height={12}
                    />
                  </div>
                ))}
              </div>
            )}
          </Panel>

          <Panel title="Incoming laps">
            {feed.length === 0 ? (
              <div className="py-4 text-[12px] text-ink-faint">No laps yet.</div>
            ) : (
              <div className="space-y-1">
                {feed.map((frame, i) => (
                  <div
                    key={`${frame.index}-${i}`}
                    className="flex items-center justify-between border-t border-line/60 py-1 text-[11.5px]"
                    style={{ opacity: 1 - i * 0.09 }}
                  >
                    <span className="num">
                      L{frame.observation?.session_lap} {frame.observation?.driver}
                    </span>
                    <span className="num text-ink-dim">
                      {frame.observation?.lap_time.toFixed(3)}s
                    </span>
                    <span
                      className="num text-[11px]"
                      style={{
                        color:
                          Math.abs(frame.state?.innovation_z ?? 0) > 2.5
                            ? 'var(--color-alert)'
                            : 'var(--color-ink-faint)',
                      }}
                      title="prediction error, in standard deviations"
                    >
                      z {signed(frame.state?.innovation_z ?? 0, 1)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  )
}
