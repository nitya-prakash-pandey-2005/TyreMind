/**
 * Typed client for the TyreMind API.
 *
 * Every estimate the backend returns carries a standard deviation, and the
 * types make that non-optional. It is deliberately hard to render a point
 * estimate here without also having the spread to hand.
 */

export interface SessionRef {
  session_id: string
  year: number
  grand_prix: string
  session: string
  label: string
  cached: boolean
}

export interface CompoundEstimate {
  degradation_rate: number
  degradation_rate_sd: number
  ci95: [number, number]
  naive_estimate: number | null
  laps: number
}

export interface ConfounderEstimate {
  mean: number
  sd: number
  prior_mean?: number
  prior_sd?: number
  note: string
}

export interface SessionSummary {
  session: SessionRef
  quality: {
    session_name?: string
    total_laps?: number
    retained_laps?: number
    exclusions?: Record<string, number>
    retention_rate?: number
    quality_score?: number
    longest_run?: number
    n_drivers?: number
  }
  n_laps: number
  n_drivers: number
  n_runs: number
  compounds: Record<string, CompoundEstimate>
  confounders: {
    fuel_slope: ConfounderEstimate
    track_evolution: ConfounderEstimate
    traffic: ConfounderEstimate
  }
  diagnostics: {
    loglik: number
    aic: number
    bic: number
    converged: boolean
    observation_noise_sd: number
    n_states: number
  }
}

export interface RunRow {
  driver: string
  run_id: number
  compound: string
  laps: number
  first_lap: number
  last_lap: number
  start_age: number
  end_age: number
  median_lap_time: number
}

export interface Contribution {
  key: string
  label: string
  seconds: number
  sd: number
  ci95: [number, number]
  is_tyre: boolean
}

export interface Decomposition {
  driver: string
  session_lap: number
  reference_lap: number
  observed_delta: number
  residual: number
  tyre_age: number
  compound: string
  tyre_seconds: number
  confounder_seconds: number
  tyre_share: number
  contributions: Contribution[]
}

export interface DecompositionRow {
  session_lap: number
  tyre_age: number
  compound: string
  observed_delta: number
  residual: number
  tyre?: number
  fuel?: number
  track?: number
  traffic?: number
}

export interface Scenario {
  scenario: string
  label: string
  actual_lap_time: number
  estimated_lap_time: number
  delta: number
  sd: number
  ci95: [number, number]
  note: string
}

export interface ProjectionResult {
  driver: string
  from_lap: number
  tyre_age: number
  compound: string
  threshold_s: number
  competitive_life_laps: number
  competitive_life_lower: number
  competitive_life_upper: number
  horizon: number[]
  loss: number[]
  loss_sd: number[]
  breach_probability: number[]
  applicability: number[]
}

export interface DegradationRow {
  driver: string
  session_lap: number
  run_id: number
  compound: string
  tyre_age: number
  level: number
  level_sd: number
  rate: number
  rate_sd: number
}

export interface LiveState {
  driver: string
  session_lap: number
  compound: string
  tyre_age: number
  performance_loss: number
  performance_loss_sd: number
  degradation_rate: number
  degradation_rate_sd: number
  health_index: number
  laps_observed: number
  innovation: number
  innovation_z: number
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) {
    const body = await response.text()
    let detail = body
    try {
      detail = JSON.parse(body).detail ?? body
    } catch {
      /* body was not JSON; use it as-is */
    }
    throw new Error(detail || `${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => get<{ status: string; sessions_cached: number; offline_ready: boolean }>('/api/health'),
  sessions: () => get<SessionRef[]>('/api/sessions'),
  summary: (id: string) => get<SessionSummary>(`/api/session/${id}`),
  runs: (id: string) => get<RunRow[]>(`/api/session/${id}/runs`),
  degradation: (id: string, smoothed = true) =>
    get<{ estimate_type: string; rows: DegradationRow[] }>(
      `/api/session/${id}/degradation?smoothed=${smoothed}`,
    ),
  track: (id: string) =>
    get<{ rows: { session_lap: number; track_effect: number; track_effect_sd: number }[] }>(
      `/api/session/${id}/track`,
    ),
  decompose: (id: string, driver: string, lap: number, referenceLap?: number) =>
    get<Decomposition>(
      `/api/session/${id}/decompose?driver=${encodeURIComponent(driver)}&lap=${lap}` +
        (referenceLap != null ? `&reference_lap=${referenceLap}` : ''),
    ),
  decomposeRun: (id: string, driver: string, runId: number) =>
    get<{ rows: DecompositionRow[] }>(
      `/api/session/${id}/decompose-run?driver=${encodeURIComponent(driver)}&run_id=${runId}`,
    ),
  counterfactual: (id: string, driver: string, lap: number) =>
    get<{ scenarios: Scenario[]; disclaimer: string }>(
      `/api/session/${id}/counterfactual?driver=${encodeURIComponent(driver)}&lap=${lap}`,
    ),
  projection: (id: string, driver: string, lap: number, horizon = 20) =>
    get<ProjectionResult>(
      `/api/session/${id}/projection?driver=${encodeURIComponent(driver)}&lap=${lap}&horizon=${horizon}`,
    ),
  experiments: () => get<Record<string, any>>('/api/experiments'),
}

/** Pirelli compound band colours, used wherever a compound is shown. */
export const COMPOUND_COLOUR: Record<string, string> = {
  SOFT: 'var(--color-soft)',
  MEDIUM: 'var(--color-medium)',
  HARD: 'var(--color-hard)',
  INTERMEDIATE: 'var(--color-intermediate)',
  WET: 'var(--color-wet)',
}

/** Confounders are cool by design, so the tyre reads warm against them. */
export const TERM_COLOUR: Record<string, string> = {
  tyre: 'var(--color-alert)',
  fuel: 'var(--color-fuel)',
  track: 'var(--color-track)',
  traffic: 'var(--color-traffic)',
  residual: 'var(--color-residual)',
}

export function compoundColour(compound: string): string {
  return COMPOUND_COLOUR[compound?.toUpperCase()] ?? 'var(--color-ink-dim)'
}

/** Format seconds with an explicit sign. In a timing context the sign is the message. */
export function signed(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return '—'
  return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`
}

export function fixed(value: number | null | undefined, digits = 3): string {
  return value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits)
}
