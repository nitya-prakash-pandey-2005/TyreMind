/**
 * TyreMind dashboard shell.
 *
 * Laid out as an instrument rather than a page: a fixed session rail on the
 * left, a thin status header, and one working area. Views are tabs across the
 * same session, because a race engineer switching between "what is happening"
 * and "why" should not lose their place.
 */

import { useEffect, useState } from 'react'
import {
  api,
  compoundColour,
  fixed,
  signed,
  type Decomposition,
  type DecompositionRow,
  type ProjectionResult,
  type RunRow,
  type Scenario,
  type SessionRef,
  type SessionSummary,
} from './lib/api'
import { Beam, CompoundChip, Empty, ErrorNote, Loading, Panel, Stat, EstimateTag } from './components/primitives'
import { PeelAway } from './components/PeelAway'
import { Waterfall } from './components/Waterfall'
import { LiveMonitor } from './components/LiveMonitor'
import { SciencePanel } from './components/SciencePanel'

type View = 'explain' | 'live' | 'twin' | 'science'

const VIEWS: { key: View; label: string; blurb: string }[] = [
  { key: 'explain', label: 'Why is the car slow', blurb: 'Attribute pace to its causes' },
  { key: 'live', label: 'Live monitor', blurb: 'Estimate updating lap by lap' },
  { key: 'twin', label: 'Tyre state', blurb: 'Health, projection, what-ifs' },
  { key: 'science', label: 'Method', blurb: 'How it was validated' },
]

export default function App() {
  const [sessions, setSessions] = useState<SessionRef[]>([])
  const [sessionId, setSessionId] = useState('')
  const [view, setView] = useState<View>('explain')
  const [offline, setOffline] = useState<boolean | null>(null)
  const [bootError, setBootError] = useState('')

  useEffect(() => {
    Promise.all([api.sessions(), api.health()])
      .then(([list, health]) => {
        setSessions(list)
        setOffline(health.offline_ready)
        const race = list.find((s) => s.session === 'R') ?? list[0]
        if (race) setSessionId(race.session_id)
      })
      .catch((e) => setBootError(String(e.message ?? e)))
  }, [])

  if (bootError) {
    return (
      <div className="mx-auto max-w-lg p-10">
        <ErrorNote error={`${bootError}. Start the API with: python -m tyremind.serve`} />
      </div>
    )
  }

  const current = sessions.find((s) => s.session_id === sessionId)

  return (
    <div className="flex min-h-screen flex-col">
      <header className="flex shrink-0 flex-wrap items-center gap-x-5 gap-y-2 border-b border-line px-4 py-2.5">
        <div className="flex items-baseline gap-2.5">
          <span className="text-[15px] font-bold tracking-[-0.02em]">TYREMIND</span>
          <span className="hidden text-[11px] text-ink-faint sm:inline">
            observed pace is not tyre degradation
          </span>
        </div>

        <div className="ml-auto flex items-center gap-4 text-[11px] text-ink-faint">
          {current && (
            <span className="num text-ink-dim">
              {current.year} {current.grand_prix} · {current.session}
            </span>
          )}
          {offline != null && (
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-1.5 w-1.5 rounded-full"
                style={{ background: offline ? 'var(--color-good)' : 'var(--color-medium)' }}
              />
              {offline ? 'offline ready' : 'needs network'}
            </span>
          )}
        </div>
      </header>

      <div className="flex flex-1 flex-col lg:flex-row">
        <nav className="shrink-0 border-b border-line lg:w-56 lg:border-r lg:border-b-0">
          <div className="px-3 py-3">
            <div className="mb-1.5 text-[10px] text-ink-faint">Session</div>
            <div className="space-y-px">
              {sessions.map((s) => (
                <button
                  key={s.session_id}
                  onClick={() => setSessionId(s.session_id)}
                  className={`block w-full px-2 py-1.5 text-left text-[12px] transition-colors ${
                    s.session_id === sessionId
                      ? 'bg-raised text-ink'
                      : 'text-ink-dim hover:bg-raised/50 hover:text-ink'
                  }`}
                >
                  <span className="num">{s.grand_prix}</span>
                  <span className="ml-1.5 text-[10px] text-ink-faint">{s.session}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="border-t border-line px-3 py-3">
            <div className="mb-1.5 text-[10px] text-ink-faint">View</div>
            <div className="space-y-px">
              {VIEWS.map((v) => (
                <button
                  key={v.key}
                  onClick={() => setView(v.key)}
                  className={`block w-full px-2 py-1.5 text-left transition-colors ${
                    v.key === view ? 'bg-raised' : 'hover:bg-raised/50'
                  }`}
                >
                  <span
                    className={`block text-[12px] ${
                      v.key === view ? 'font-medium text-alert' : 'text-ink-dim'
                    }`}
                  >
                    {v.label}
                  </span>
                  <span className="block text-[10px] text-ink-faint">{v.blurb}</span>
                </button>
              ))}
            </div>
          </div>
        </nav>

        <main className="min-w-0 flex-1 p-3">
          {!sessionId ? (
            <Loading what="the session catalogue" />
          ) : view === 'live' ? (
            <LiveMonitor sessionId={sessionId} />
          ) : view === 'science' ? (
            <SciencePanel sessionId={sessionId} />
          ) : view === 'twin' ? (
            <TyreStateView sessionId={sessionId} />
          ) : (
            <ExplainView sessionId={sessionId} />
          )}
        </main>
      </div>
    </div>
  )
}

/** Shared run picker: a degradation conclusion is drawn from a run, not a driver. */
function useRuns(sessionId: string) {
  const [runs, setRuns] = useState<RunRow[]>([])
  const [selected, setSelected] = useState<RunRow | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setRuns([])
    setSelected(null)
    api
      .runs(sessionId)
      .then((rows) => {
        const usable = rows.filter((r) => r.laps >= 5)
        setRuns(usable)
        setSelected(usable[0] ?? null)
      })
      .catch((e) => setError(String(e.message ?? e)))
  }, [sessionId])

  return { runs, selected, setSelected, error }
}

function RunPicker({
  runs,
  selected,
  onSelect,
}: {
  runs: RunRow[]
  selected: RunRow | null
  onSelect: (r: RunRow) => void
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {runs.slice(0, 14).map((run) => {
        const active = selected?.run_id === run.run_id
        return (
          <button
            key={run.run_id}
            onClick={() => onSelect(run)}
            className={`flex items-center gap-1.5 border px-2 py-1 text-[11px] transition-colors ${
              active ? 'border-alert text-ink' : 'border-line text-ink-dim hover:border-line-bright'
            }`}
          >
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: compoundColour(run.compound) }}
            />
            <span className="num">{run.driver}</span>
            <span className="text-ink-faint">{run.laps}L</span>
          </button>
        )
      })}
    </div>
  )
}

function ExplainView({ sessionId }: { sessionId: string }) {
  const { runs, selected, setSelected, error } = useRuns(sessionId)
  const [summary, setSummary] = useState<SessionSummary | null>(null)
  const [rows, setRows] = useState<DecompositionRow[]>([])
  const [decomposition, setDecomposition] = useState<Decomposition | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    setSummary(null)
    api.summary(sessionId).then(setSummary).catch(() => undefined)
  }, [sessionId])

  useEffect(() => {
    if (!selected) return
    setBusy(true)
    Promise.all([
      api.decomposeRun(sessionId, selected.driver, selected.run_id),
      api.decompose(sessionId, selected.driver, selected.last_lap),
    ])
      .then(([run, lap]) => {
        setRows(run.rows)
        setDecomposition(lap)
      })
      .catch(() => undefined)
      .finally(() => setBusy(false))
  }, [sessionId, selected])

  if (error) return <ErrorNote error={error} />
  if (!selected) return <Loading what="the session" />

  return (
    <div className="space-y-3">
      <Panel
        title="Peel the confounders away"
        aside={`${selected.driver} · ${selected.laps} laps on ${selected.compound}`}
      >
        <div className="mb-4">
          <RunPicker runs={runs} selected={selected} onSelect={setSelected} />
        </div>
        {busy || rows.length === 0 ? (
          <Loading what="this run" />
        ) : (
          <PeelAway rows={rows} compound={selected.compound} />
        )}
      </Panel>

      <div className="grid gap-3 lg:grid-cols-[1.3fr_1fr]">
        <Panel title="Where the lap time went">
          {decomposition ? <Waterfall decomposition={decomposition} /> : <Loading what="the lap" />}
        </Panel>

        {summary && <ConfounderPanel summary={summary} />}
      </div>
    </div>
  )
}

function ConfounderPanel({ summary }: { summary: SessionSummary }) {
  const { fuel_slope, track_evolution, traffic } = summary.confounders
  return (
    <div className="space-y-3">
      <Panel title="Degradation by compound" aside="s/lap">
        <div className="space-y-3.5">
          {Object.entries(summary.compounds).map(([compound, est]) => (
            <div key={compound}>
              <div className="mb-1 flex items-baseline justify-between">
                <CompoundChip compound={compound} />
                <span className="num text-[13px]">
                  {fixed(est.degradation_rate)}
                  <span className="ml-1 text-[10px] text-ink-faint">
                    ± {est.degradation_rate_sd.toFixed(3)}
                  </span>
                </span>
              </div>
              <Beam
                mean={est.degradation_rate}
                sd={est.degradation_rate_sd}
                domain={[-0.08, 0.28]}
                colour={compoundColour(compound)}
                zero
                height={14}
              />
              {est.naive_estimate != null && (
                <div className="mt-1 text-[10.5px] text-ink-faint">
                  lap-time-vs-tyre-age would say{' '}
                  <span className="num" style={{ color: 'var(--color-ink-dim)' }}>
                    {fixed(est.naive_estimate)}
                  </span>
                  {est.naive_estimate < 0 && ' — i.e. the tyre getting faster with age'}
                </div>
              )}
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="What else moved the lap time" aside="session estimates">
        <div className="space-y-3.5 text-[12px]">
          <ConfounderRow
            label="Fuel burn-off"
            value={`${fixed(fuel_slope.mean)} s/lap`}
            detail={`prior ${fuel_slope.prior_mean?.toFixed(3)} ± ${fuel_slope.prior_sd?.toFixed(3)}`}
            note={fuel_slope.note}
            colour="var(--color-fuel)"
          />
          <ConfounderRow
            label="Track evolution"
            value={`${fixed(track_evolution.mean, 2)} s over session`}
            detail={`± ${track_evolution.sd.toFixed(2)}`}
            note={track_evolution.note}
            colour="var(--color-track)"
          />
          <ConfounderRow
            label="Traffic"
            value={`${fixed(traffic.mean, 2)} s at worst`}
            detail={`± ${traffic.sd.toFixed(2)}`}
            note={traffic.note}
            colour="var(--color-traffic)"
          />
        </div>
      </Panel>
    </div>
  )
}

function ConfounderRow({
  label,
  value,
  detail,
  note,
  colour,
}: {
  label: string
  value: string
  detail: string
  note: string
  colour: string
}) {
  return (
    <div className="border-l-2 pl-3" style={{ borderColor: colour }}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-ink">{label}</span>
        <span className="num text-[12.5px] text-ink">{value}</span>
      </div>
      <div className="num text-[10.5px] text-ink-faint">{detail}</div>
      <p className="mt-1 max-w-[46ch] text-[11px] leading-relaxed text-ink-faint">{note}</p>
    </div>
  )
}

function TyreStateView({ sessionId }: { sessionId: string }) {
  const { runs, selected, setSelected, error } = useRuns(sessionId)
  const [projection, setProjection] = useState<ProjectionResult | null>(null)
  const [scenarios, setScenarios] = useState<Scenario[]>([])

  useEffect(() => {
    if (!selected) return
    const lap = Math.round((selected.first_lap + selected.last_lap) / 2)
    Promise.all([
      api.projection(sessionId, selected.driver, lap, 18),
      api.counterfactual(sessionId, selected.driver, lap),
    ])
      .then(([p, c]) => {
        setProjection(p)
        setScenarios(c.scenarios)
      })
      .catch(() => undefined)
  }, [sessionId, selected])

  if (error) return <ErrorNote error={error} />
  if (!selected) return <Loading what="the session" />

  return (
    <div className="space-y-3">
      <Panel title="Select a tyre">
        <RunPicker runs={runs} selected={selected} onSelect={setSelected} />
      </Panel>

      {!projection ? (
        <Loading what="the projection" />
      ) : (
        <div className="grid gap-3 lg:grid-cols-[1fr_1fr]">
          <Panel
            title="Remaining competitive life"
            aside={`threshold ${projection.threshold_s} s/lap`}
          >
            <div className="mb-5 grid grid-cols-3 gap-4">
              <Stat
                label="Tyre age"
                value={projection.tyre_age.toFixed(0)}
                unit="laps"
              />
              <Stat
                label="Expected life left"
                value={projection.competitive_life_laps.toFixed(0)}
                unit="laps"
                tone="warm"
                hint={`between ${projection.competitive_life_lower.toFixed(0)} and ${projection.competitive_life_upper.toFixed(0)}`}
              />
              <Stat label="Compound" value={projection.compound} tone="dim" />
            </div>

            <div className="space-y-1.5">
              <div className="grid grid-cols-[52px_1fr_54px] gap-2 text-[10px] text-ink-faint">
                <span>ahead</span>
                <span>probability past the threshold</span>
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
                    className="grid grid-cols-[52px_1fr_54px] items-center gap-2 border-t border-line/60 py-1"
                  >
                    <span className="num text-[11.5px] text-ink-dim">+{h}</span>
                    <div className="flex items-center gap-2">
                      <div className="h-3 flex-1 bg-raised">
                        <div
                          className="h-full transition-[width]"
                          style={{
                            width: `${p * 100}%`,
                            background: compoundColour(projection.compound),
                            opacity: 0.55 + 0.45 * applies,
                          }}
                        />
                      </div>
                      <span className="num w-9 text-right text-[11.5px]">
                        {(p * 100).toFixed(0)}%
                      </span>
                    </div>
                    <span
                      className="num text-right text-[10.5px]"
                      style={{
                        color: applies < 0.5 ? 'var(--color-alert)' : 'var(--color-ink-faint)',
                      }}
                    >
                      {(applies * 100).toFixed(0)}%
                    </span>
                  </div>
                )
              })}
            </div>
            <p className="mt-3 max-w-[54ch] text-[11px] leading-relaxed text-ink-faint">
              The right-hand column falls once the projection runs past the oldest tyre age
              this session actually contains. Below about 50% the model is extrapolating a
              trend rather than reporting one.
            </p>
          </Panel>

          <Panel title="What if" aside={<EstimateTag>never driven</EstimateTag>}>
            {scenarios.length === 0 ? (
              <Empty>Select a tyre to run scenarios.</Empty>
            ) : (
              <div className="space-y-4">
                {scenarios.map((s) => (
                  <div key={s.scenario} className="border-t border-line pt-3 first:border-0 first:pt-0">
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="text-[12.5px] text-ink">{s.label}</span>
                      <span
                        className="num text-[15px]"
                        style={{
                          color: s.delta < 0 ? 'var(--color-good)' : 'var(--color-ink-dim)',
                        }}
                      >
                        {signed(s.delta)} s
                      </span>
                    </div>
                    <div className="mt-1.5">
                      <Beam
                        mean={s.delta}
                        sd={s.sd}
                        domain={[-2, 0.5]}
                        colour={s.delta < 0 ? 'var(--color-good)' : 'var(--color-residual)'}
                        zero
                        height={12}
                      />
                    </div>
                    <p className="mt-1.5 max-w-[50ch] text-[11px] leading-relaxed text-ink-faint">
                      {s.note}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </Panel>
        </div>
      )}
    </div>
  )
}
