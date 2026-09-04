/**
 * The tyre digital twin: a car seen from above, with each corner drawn as what
 * the model believes about it.
 *
 * This is the one place the physics layer becomes something you can look at.
 * Curvature recovered from GPS traces gives lateral acceleration, load transfer
 * splits that across the four corners, and the result is that a clockwise
 * circuit visibly works its left-hand tyres harder. The asymmetry on screen is
 * measured, not decorative -- `exp06_circuit_asymmetry` recovers circuit
 * rotation direction from it on 7 of 8 circuits.
 *
 * Drawn as inline SVG rather than a chart library because the geometry *is* the
 * information: where a tyre sits on the car is what the reader needs to map the
 * number onto the real object.
 */

import { compoundColour } from '../lib/api'

export interface CornerLoad {
  FL: number
  FR: number
  RL: number
  RR: number
}

function healthColour(health: number): string {
  if (health > 60) return 'var(--color-good)'
  if (health > 25) return 'var(--color-medium)'
  return 'var(--color-alert)'
}

const CORNERS: { key: keyof CornerLoad; label: string; x: number; y: number }[] = [
  { key: 'FL', label: 'front left', x: 46, y: 44 },
  { key: 'FR', label: 'front right', x: 134, y: 44 },
  { key: 'RL', label: 'rear left', x: 46, y: 176 },
  { key: 'RR', label: 'rear right', x: 134, y: 176 },
]

/**
 * @param energy Per-corner energy share. Values are relative; they are
 *   normalised here so the drawing reads the same whatever units arrive.
 * @param health 0-100 tyre health, used for the fill colour.
 * @param compound Compound in use, for the band colour.
 */
export function TyreTwin({
  energy,
  health,
  performanceLost,
  compound,
  ageLaps,
}: {
  energy: CornerLoad
  /** Null while the timeline is still loading. Never defaulted to 100 -- showing
   *  a fresh-tyre reading for a 40-lap-old set is worse than showing nothing. */
  health: number | null
  performanceLost: number | null
  compound: string
  ageLaps: number
}) {
  const values = Object.values(energy)
  const max = Math.max(...values, 1e-9)
  const min = Math.min(...values)
  const span = Math.max(max - min, 1e-9)

  const band = compoundColour(compound)
  const total = values.reduce((a, b) => a + b, 0) || 1
  const leftShare = (energy.FL + energy.RL) / total
  const frontShare = (energy.FL + energy.FR) / total

  return (
    <div className="flex flex-col items-center gap-3 sm:flex-row sm:items-start sm:gap-6">
      <svg viewBox="0 0 180 230" className="h-[230px] w-[180px] shrink-0" role="img"
        aria-label={`Tyre loading diagram: left side carries ${(leftShare * 100).toFixed(0)} percent of energy`}>
        {/* Car body, kept deliberately plain so the tyres carry the attention. */}
        <path
          d="M 90 10 C 70 10 62 28 62 48 L 62 172 C 62 194 72 214 90 214 C 108 214 118 194 118 172 L 118 48 C 118 28 110 10 90 10 Z"
          fill="var(--color-raised)"
          stroke="var(--color-line)"
          strokeWidth="1"
        />
        {/* Axles */}
        <line x1="52" y1="44" x2="128" y2="44" stroke="var(--color-line)" strokeWidth="1" />
        <line x1="52" y1="176" x2="128" y2="176" stroke="var(--color-line)" strokeWidth="1" />

        {CORNERS.map(({ key, x, y }) => {
          const value = energy[key]
          // Intensity is relative WITHIN this car, so the hardest-worked corner
          // always reads as the hardest-worked corner regardless of absolute scale.
          const intensity = (value - min) / span
          const width = 17
          const height = 34
          return (
            <g key={key}>
              <rect
                x={x - width / 2}
                y={y - height / 2}
                width={width}
                height={height}
                rx="3"
                fill={`color-mix(in oklab, ${band} ${18 + intensity * 62}%, var(--color-ground))`}
                stroke={band}
                strokeWidth="1.2"
              />
              <text
                x={x}
                y={y + 4}
                textAnchor="middle"
                className="num"
                style={{ fontSize: 9, fill: 'var(--color-ink)' }}
              >
                {(100 * (value / total)).toFixed(0)}
              </text>
              <text
                x={x}
                y={y - height / 2 - 5}
                textAnchor="middle"
                style={{ fontSize: 8, fill: 'var(--color-ink-faint)' }}
              >
                {key}
              </text>
            </g>
          )
        })}

        <text x="90" y="226" textAnchor="middle" style={{ fontSize: 8.5, fill: 'var(--color-ink-faint)' }}>
          % of frictional energy per corner
        </text>
      </svg>

      <div className="min-w-0 flex-1 space-y-3">
        <div>
          <div className="text-[11px] text-ink-faint">Tyre health</div>
          {health == null ? (
            <div className="num text-[34px] leading-none font-medium text-ink-faint">--</div>
          ) : (
            <>
              <div className="flex items-baseline gap-2">
                <span
                  className="num text-[34px] leading-none font-medium"
                  style={{ color: healthColour(health) }}
                >
                  {health.toFixed(0)}
                </span>
                <span className="text-[12px] text-ink-faint">/ 100</span>
              </div>
              {performanceLost != null && (
                <div className="num mt-0.5 text-[11px] text-ink-faint">
                  {performanceLost.toFixed(2)} s/lap slower than a fresh set
                </div>
              )}
              <div className="mt-1.5 h-1.5 w-full max-w-[240px] bg-raised">
                <div
                  className="h-full transition-[width] duration-500"
                  style={{ width: `${health}%`, background: healthColour(health) }}
                />
              </div>
            </>
          )}
        </div>

        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-[11.5px]">
          <div>
            <dt className="text-ink-faint">Compound</dt>
            <dd className="flex items-center gap-1.5 text-ink">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ background: band }}
              />
              {compound}
            </dd>
          </div>
          <div>
            <dt className="text-ink-faint">Age</dt>
            <dd className="num text-ink">{ageLaps.toFixed(0)} laps</dd>
          </div>
          <div>
            <dt className="text-ink-faint">Left / right split</dt>
            <dd className="num text-ink">
              {(leftShare * 100).toFixed(0)} / {((1 - leftShare) * 100).toFixed(0)}
            </dd>
          </div>
          <div>
            <dt className="text-ink-faint">Front / rear split</dt>
            <dd className="num text-ink">
              {(frontShare * 100).toFixed(0)} / {((1 - frontShare) * 100).toFixed(0)}
            </dd>
          </div>
        </dl>

        <p className="max-w-[46ch] text-[11px] leading-relaxed text-ink-faint">
          Corner loading is computed from the racing line: curvature gives lateral
          acceleration, which transfers load to the outside of the car. A circuit
          with mostly right-hand corners works the left tyres hardest.
        </p>
      </div>
    </div>
  )
}

/**
 * Per-corner energy shares come from the API, which reads them from the recorded
 * physics validation rather than reconstructing them.
 *
 * The dashboard runs from cached timing data, which does not carry the position
 * telemetry the physics layer needs -- that is a far larger download, used by the
 * experiments. So the twin shows measured values for circuits that have been
 * analysed and says plainly when a circuit has not, rather than filling the gap
 * with an even split presented as a result.
 */
export interface CornerEnergyResult {
  circuit: string
  measured: boolean
  reason?: string
  corner_share?: CornerLoad
  left_side_energy_share?: number
  front_axle_energy_share?: number
  peak_lateral_g?: number
  published_direction?: string
  predicted_direction?: string
  n_laps?: number
}

export const EVEN_SPLIT: CornerLoad = { FL: 0.25, FR: 0.25, RL: 0.25, RR: 0.25 }
