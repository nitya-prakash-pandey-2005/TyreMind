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

// ---------------------------------------------------------------------------
// Strategy, trust, business value and cross-industry
// ---------------------------------------------------------------------------

export interface StrategyOption {
  label: string
  pit_lap: number | null
  new_compound: string | null
  expected_time: number
  time_sd: number
  best_case: number
  downside: number
  ran_out_of_tyre: number
  n_sims: number
}

export interface Distribution {
  centres: number[]
  counts: number[]
  mean: number
  p10: number
  p90: number
}

export interface StrategyResult {
  recommended: string
  margin_s: number
  decision_confidence: number
  reasons: string[]
  alternatives: StrategyOption[]
  state: {
    current_lap: number
    total_laps: number
    laps_remaining: number
    compound: string
    tyre_age: number
    pit_loss_s: number
  }
  narration: { text: string; source: string }
  distributions: Record<string, Distribution>
}

export interface ConsensusEntry {
  compound: string
  estimates: Record<string, { mean: number; sd: number }>
  consensus: number
  consensus_sd: number
  spread: number
  agreement: number
  disagreement_flagged: boolean
  explanation: string
}

export interface TrustResult {
  consensus: Record<string, ConsensusEntry>
  applicability: {
    applicability: number
    risk: string
    reasons: string[]
    checks: Record<string, Record<string, number>>
  }
  regimes: {
    driver: string
    run_id: number
    laps: number
    regime: string
    confidence: number
    meaning: string
  }[]
  value_of_information: {
    signal: string
    current_uncertainty: number
    estimated_reduction: number
    projected_uncertainty: number
    rationale: string
  }[]
}

export interface ValueEstimate {
  metric: string
  value: number
  unit: string
  derivation: string
  source_quantity: string
  confidence: 'measured' | 'estimated' | 'illustrative'
}

export interface BusinessReport {
  estimates: ValueEstimate[]
  caveats: string[]
}

export interface AssetProfileDto {
  asset_type: string
  display_name: string
  age_unit: string
  performance_unit: string
  confounders: string[]
  typical_life: number
  performance_threshold: number
  notes: string
}

export interface CrossIndustryResult {
  profiles: AssetProfileDto[]
  validated_transfer: {
    dataset: string
    n_engines_scored: number
    n_sensors_used: number
    rul_rmse: number
    rul_mae: number
    fraction_early: number
    estimated_degradation_rate: number
    /** Per-engine predictions and labels, for the validation scatter. */
    predictions?: number[]
    truths?: number[]
    note: string
  } | null
  fleet_illustration: BusinessReport
  honest_summary: string
}

export interface ValidationResult {
  overall: {
    n_events: number
    n_comparisons: number
    mae: number
    naive_mae: number | null
    bias: number
    coverage_95: number
  }
  this_event: unknown
  all_events: {
    event: string
    mae: number
    naive_mae: number
    bias: number
    coverage_95: number
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

export interface HealthTimeline {
  driver: string
  run_id: number
  compound: string
  rows: {
    session_lap: number
    tyre_age: number
    level: number
    level_sd: number
    rate: number
    rate_sd: number
    health: number
  }[]
  health_anchor_note: string
}

export interface Narration {
  text: string
  source: string
  facts: Record<string, unknown>
  rejected_reason: string | null
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) {
    const body = await response.text()
    let detail = body
    try {
      detail = JSON.parse(body).detail ?? body
    } catch {
      /* not JSON */
    }
    throw new Error(detail || `${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export interface PitWindow {
  driver: string
  from_lap: number
  total_laps: number
  new_compound: string
  stay_out_expected_time: number
  optimum_lap: number
  optimum_expected_time: number
  window_within_1s: [number, number] | null
  sweep: {
    pit_lap: number
    expected_time: number
    downside: number
    best_case: number
    runs_past_cliff: number
  }[]
  n_sims: number
  note: string
}

export const advanced = {
  pitWindow: (id: string, driver: string, lap: number, nSims = 1200) =>
    getJson<PitWindow>(
      `/api/session/${id}/pit-window?driver=${encodeURIComponent(driver)}&lap=${lap}&n_sims=${nSims}`,
    ),
  strategy: (id: string, driver: string, lap: number, nSims = 5000) =>
    getJson<StrategyResult>(
      `/api/session/${id}/strategy?driver=${encodeURIComponent(driver)}&lap=${lap}&n_sims=${nSims}`,
    ),
  regret: (id: string, driver: string, lap: number, recommended: number, actual: number) =>
    getJson<{ regret_s: number; recommended_expected_time: number; actual_expected_time: number }>(
      `/api/session/${id}/regret?driver=${encodeURIComponent(driver)}&lap=${lap}` +
        `&recommended_lap=${recommended}&actual_lap=${actual}`,
    ),
  trust: (id: string, compound?: string, tyreAge = 20) =>
    getJson<TrustResult>(
      `/api/session/${id}/trust?tyre_age=${tyreAge}` +
        (compound ? `&compound=${encodeURIComponent(compound)}` : ''),
    ),
  narrate: (id: string, driver: string, lap: number) =>
    getJson<{ decomposition: Narration; projection: Narration }>(
      `/api/session/${id}/narrate?driver=${encodeURIComponent(driver)}&lap=${lap}`,
    ),
  business: () => getJson<BusinessReport>('/api/business'),
  crossIndustry: () => getJson<CrossIndustryResult>('/api/cross-industry'),
  validation: (id: string) => getJson<ValidationResult>(`/api/session/${id}/validation`),
  healthTimeline: (id: string, driver: string, runId: number) =>
    getJson<HealthTimeline>(
      `/api/session/${id}/health-timeline?driver=${encodeURIComponent(driver)}&run_id=${runId}`,
    ),
}

export interface CornerEnergy {
  circuit: string
  measured: boolean
  reason?: string
  corner_share?: { FL: number; FR: number; RL: number; RR: number }
  left_side_energy_share?: number
  front_axle_energy_share?: number
  peak_lateral_g?: number
  published_direction?: string
  predicted_direction?: string
  n_laps?: number
}

export function cornerEnergy(circuit: string): Promise<CornerEnergy> {
  return getJson<CornerEnergy>(`/api/physics/corner-energy?circuit=${encodeURIComponent(circuit)}`)
}

/**
 * Resolve a CSS custom property to a concrete colour.
 *
 * ECharts draws to canvas and cannot resolve `var(--x)` -- it silently falls back
 * to its own default palette, which is how every chart in this app ended up
 * monochrome grey while the surrounding HTML was correctly themed. Anything
 * handed to a chart has to go through here first.
 *
 * @param value A CSS colour, which may be `var(--name)` or already concrete.
 * @returns A concrete colour string usable by canvas.
 */
export function resolveColour(value: string): string {
  const match = /^var\((--[\w-]+)\)$/.exec(value.trim())
  if (!match) return value
  if (typeof window === 'undefined') return FALLBACK_COLOUR[match[1]] ?? '#8fa3ae'

  const resolved = getComputedStyle(document.documentElement)
    .getPropertyValue(match[1])
    .trim()
  return resolved || FALLBACK_COLOUR[match[1]] || '#8fa3ae'
}

/** Used before the stylesheet is available, and in non-browser contexts. */
const FALLBACK_COLOUR: Record<string, string> = {
  '--color-soft': '#e8352e',
  '--color-medium': '#f5c518',
  '--color-hard': '#ededed',
  '--color-alert': '#ff8a5b',
  '--color-good': '#4bbf8a',
  '--color-fuel': '#4fa8c5',
  '--color-track': '#7b8fa1',
  '--color-traffic': '#b47fd0',
  '--color-residual': '#5a6b76',
  '--color-ink-dim': '#8fa3ae',
}

/** Compound colour, already resolved for canvas rendering. */
export function compoundColourResolved(compound: string): string {
  return resolveColour(compoundColour(compound))
}
