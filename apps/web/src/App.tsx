/**
 * TyreMind dashboard shell.
 *
 * Laid out as an instrument rather than a page: a fixed session rail on the
 * left, a thin status header, and one working area. The view order follows the
 * question a reader actually asks -- what is this, why is the car slow, what
 * state is the tyre in, what should we do, is it working live, do I believe it,
 * and does it work outside racing.
 */

import { useEffect, useState } from 'react'
import {
  advanced,
  api,
  compoundColour,
  type Decomposition,
  type DecompositionRow,
  type RunRow,
  type SessionRef,
  type SessionSummary,
} from './lib/api'
import { ErrorNote, Loading, Panel } from './components/primitives'
import { PeelAway } from './components/PeelAway'
import { Waterfall } from './components/Waterfall'
import { LiveMonitor } from './components/LiveMonitor'
import { SciencePanel } from './components/SciencePanel'
import { Overview } from './components/Overview'
import { StrategyView } from './components/StrategyView'
import { TyreStateView } from './components/TyreStateView'
import { BeyondRacing } from './components/BeyondRacing'
import { Explainer } from './components/Explainer'
import { StintDecomposition, TrackEvolutionChart } from './components/charts'
import { CircuitView } from './components/CircuitView'
import { AskPanel } from './components/AskPanel'
import { ThemeToggle } from './lib/theme'

type View =
  | 'overview'
  | 'explain'
  | 'circuit'
  | 'tyre'
  | 'strategy'
  | 'live'
  | 'evidence'
  | 'ask'
  | 'beyond'

const VIEWS: { key: View; label: string; blurb: string }[] = [
  { key: 'overview', label: 'Start here', blurb: 'What this is, in plain terms' },
  { key: 'explain', label: 'Why is the car slow', blurb: 'Split pace into its causes' },
  { key: 'circuit', label: 'Where it wears', blurb: '3D lap, coloured by load' },
  { key: 'tyre', label: 'Tyre twin', blurb: 'Condition, life left, what-ifs' },
  { key: 'strategy', label: 'When to pit', blurb: '5,000 simulated races' },
  { key: 'live', label: 'Live monitor', blurb: 'Updating lap by lap' },
  { key: 'evidence', label: 'Does it work', blurb: 'How it was validated' },
  { key: 'ask', label: 'Ask the method', blurb: 'Search the research corpus' },
  { key: 'beyond', label: 'Beyond racing', blurb: 'The same engine elsewhere' },
]

export default function App() {
  const [sessions, setSessions] = useState<SessionRef[]>([])
  const [sessionId, setSessionId] = useState('')
  const [view, setView] = useState<View>('overview')
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
        <ErrorNote
          error={`${bootError}. Start the server with: python -m tyremind.serve`}
        />
      </div>
    )
  }

  const current = sessions.find((s) => s.session_id === sessionId)

  return (
    <div className="flex min-h-screen flex-col lg:h-screen lg:overflow-hidden">
      <header className="z-20 flex shrink-0 flex-wrap items-center gap-x-5 gap-y-2 border-b border-line bg-ground px-4 py-2.5">
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
              {offline ? 'runs offline' : 'needs network'}
            </span>
          )}
          <ThemeToggle />
        </div>
      </header>

      <div className="flex flex-1 flex-col lg:min-h-0 lg:flex-row">
        <nav className="shrink-0 border-b border-line lg:w-56 lg:overflow-y-auto lg:border-r lg:border-b-0">
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
                  <span className="ml-1.5 text-[10px] text-ink-faint">
                    {s.session === 'R' ? 'race' : s.session}
                  </span>
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

        <main className="min-w-0 flex-1 p-3 lg:overflow-y-auto">
          {!sessionId ? (
            <Loading what="the session catalogue" />
          ) : view === 'beyond' ? (
            <BeyondRacing />
          ) : view === 'ask' ? (
            <AskPanel />
          ) : view === 'circuit' ? (
            <CircuitView circuit={current?.grand_prix ?? ''} />
          ) : view === 'live' ? (
            <LiveMonitor sessionId={sessionId} />
          ) : view === 'evidence' ? (
            <SciencePanel sessionId={sessionId} />
          ) : view === 'overview' ? (
            <Overview sessionId={sessionId} onOpenExplain={() => setView('explain')} />
          ) : (
            <RunScopedView
              key={sessionId}
              view={view}
              sessionId={sessionId}
              circuit={current?.grand_prix ?? ''}
            />
          )}
        </main>
      </div>
    </div>
  )
}

/**
 * Views that operate on a chosen run share their selection, so switching between
 * "why is the car slow" and "when to pit" keeps the same car in front of you.
 */
function RunScopedView({
  view,
  sessionId,
  circuit,
}: {
  view: View
  sessionId: string
  circuit: string
}) {
  const [runs, setRuns] = useState<RunRow[]>([])
  const [selected, setSelected] = useState<RunRow | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api
      .runs(sessionId)
      .then((rows) => {
        const usable = rows.filter((r) => r.laps >= 5)
        setRuns(usable)
        setSelected((current) =>
          current && usable.some((r) => r.run_id === current.run_id) ? current : usable[0] ?? null,
        )
      })
      .catch((e) => setError(String(e.message ?? e)))
  }, [sessionId])

  if (error) return <ErrorNote error={error} />
  if (!selected) return <Loading what="the session" />

  if (view === 'strategy') {
    return (
      <StrategyView
        sessionId={sessionId}
        runs={runs}
        selected={selected}
        onSelect={setSelected}
      />
    )
  }
  if (view === 'tyre') {
    return (
      <TyreStateView
        sessionId={sessionId}
        circuit={circuit}
        runs={runs}
        selected={selected}
        onSelect={setSelected}
      />
    )
  }
  return (
    <ExplainView
      sessionId={sessionId}
      runs={runs}
      selected={selected}
      onSelect={setSelected}
    />
  )
}

function ExplainView({
  sessionId,
  runs,
  selected,
  onSelect,
}: {
  sessionId: string
  runs: RunRow[]
  selected: RunRow
  onSelect: (r: RunRow) => void
}) {
  const [summary, setSummary] = useState<SessionSummary | null>(null)
  const [rows, setRows] = useState<DecompositionRow[]>([])
  const [decomposition, setDecomposition] = useState<Decomposition | null>(null)
  const [narration, setNarration] = useState('')
  const [track, setTrack] = useState<
    { session_lap: number; track_effect: number; track_effect_sd: number }[]
  >([])
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.summary(sessionId).then(setSummary).catch(() => undefined)
    api
      .track(sessionId)
      .then((t) => setTrack(t.rows))
      .catch(() => setTrack([]))
  }, [sessionId])

  useEffect(() => {
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

  useEffect(() => {
    advanced
      .narrate(sessionId, selected.driver, selected.last_lap)
      .then((n) => setNarration(n.decomposition.text))
      .catch(() => setNarration(''))
  }, [sessionId, selected])

  return (
    <div className="space-y-3">
      <Explainer id="explain" question="What am I looking at?">
        <p>
          A stint is a run on one set of tyres. Over a stint the lap times move,
          and this screen takes that movement apart: how much was the tyre, and
          how much was everything else.
        </p>
        <p>
          The chart below starts with the raw lap times and removes one cause at a
          time. What remains at the end is the tyre&rsquo;s own contribution — the
          curve a team actually wants when deciding a pit stop.
        </p>
      </Explainer>

      <Panel
        title="Peel the confounders away"
        aside={`${selected.driver} · ${selected.laps} laps on ${selected.compound}`}
      >
        <div className="mb-4 flex flex-wrap gap-1">
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
        {busy || rows.length === 0 ? (
          <Loading what="this run" />
        ) : (
          <PeelAway rows={rows} compound={selected.compound} />
        )}
      </Panel>

      {narration && (
        <Panel title="In plain language">
          <p className="max-w-[76ch] text-[13.5px] leading-relaxed text-ink">{narration}</p>
          <p className="mt-2 text-[10.5px] text-ink-faint">
            Generated from the model&rsquo;s own output. Every number here was computed,
            not written.
          </p>
        </Panel>
      )}

      {rows.length > 0 && (
        <Panel
          title="Where the time went, lap by lap"
          aside="the whole stint, not one lap"
        >
          <StintDecomposition rows={rows} />
          <p className="mt-2 max-w-[86ch] text-[11.5px] leading-relaxed text-ink-dim">
            Bars above the line cost time, bars below gain it. Watch the orange
            tyre bar grow while the blue fuel bar sinks: the lap where they cross
            is the moment the stint turns from getting faster to getting slower.
            The dashed line is what the stopwatch actually recorded — the sum of
            everything.
          </p>
        </Panel>
      )}

      <div className="grid gap-3 lg:grid-cols-[1.3fr_1fr]">
        <Panel title="Where the lap time went" aside="one lap, against the stint start">
          {decomposition ? <Waterfall decomposition={decomposition} /> : <Loading what="the lap" />}
        </Panel>

        {summary && <CompoundSummary summary={summary} />}
      </div>

      {track.length > 0 && (
        <Panel title="The circuit itself getting faster" aside="shared by every car">
          <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
            <TrackEvolutionChart rows={track} />
            <div className="space-y-2 text-[12.5px] leading-relaxed text-ink-dim">
              <p>
                As cars run, rubber goes down on the racing line and the circuit
                gains grip. Everyone speeds up — nothing about any individual tyre
                has changed.
              </p>
              <p>
                The curve flattens because rubber build-up saturates. That shape is
                not fitted freely: it is imposed, and it is what makes track
                evolution separable from a uniform shift in every degradation rate
                at all. The two are otherwise indistinguishable.
              </p>
              <p className="text-[11.5px] text-ink-faint">
                Which means this is the most assumption-heavy curve in the product.
                The band shows how much room the data leaves.
              </p>
            </div>
          </div>
        </Panel>
      )}
    </div>
  )
}

function CompoundSummary({ summary }: { summary: SessionSummary }) {
  return (
    <Panel title="Degradation by compound" aside="seconds lost per lap">
      <div className="space-y-4">
        {Object.entries(summary.compounds).map(([compound, estimate]) => (
          <div key={compound}>
            <div className="mb-1 flex items-baseline justify-between">
              <span className="flex items-center gap-1.5 text-[12px] text-ink-dim">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ background: compoundColour(compound) }}
                />
                {compound}
              </span>
              <span className="num text-[13px]">
                {estimate.degradation_rate.toFixed(3)}
                <span className="ml-1 text-[10px] text-ink-faint">
                  ± {estimate.degradation_rate_sd.toFixed(3)}
                </span>
              </span>
            </div>
            <div className="relative h-3.5 bg-raised">
              <div
                className="absolute top-0 bottom-0 w-px bg-line-bright"
                style={{ left: `${((0 + 0.08) / 0.36) * 100}%` }}
              />
              <div
                className="beam absolute top-0 bottom-0"
                style={{
                  left: `${Math.max(0, ((estimate.ci95[0] + 0.08) / 0.36) * 100)}%`,
                  width: `${Math.max(1, ((estimate.ci95[1] - estimate.ci95[0]) / 0.36) * 100)}%`,
                  ['--beam-color' as string]: compoundColour(compound),
                }}
              />
            </div>
            <div className="mt-1 text-[10.5px] text-ink-faint">
              Over a 25-lap stint that is{' '}
              <span className="num text-ink-dim">
                {(estimate.degradation_rate * 25).toFixed(1)} s
              </span>{' '}
              of accumulated pace loss.
            </div>
          </div>
        ))}
      </div>
    </Panel>
  )
}
