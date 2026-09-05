/**
 * The generalisation case: the same estimator on something that is not a tyre.
 *
 * This screen has one hard job -- to be persuasive without overclaiming. The
 * honest position is uncomfortable and is stated first: there is no public
 * dataset pairing tyre tread depth with vehicle telematics, so a fleet claim has
 * nothing to be scored against.
 *
 * What CAN be shown is that the identical model, pointed at NASA's turbofan
 * degradation benchmark, produces usable remaining-life predictions against
 * published ground truth. That is a real result on real data, and it is the one
 * this page leads with. The fleet arithmetic sits below it, labelled as
 * arithmetic.
 */

import { useEffect, useState } from 'react'
import { advanced, type BusinessReport, type CrossIndustryResult } from '../lib/api'
import { ErrorNote, Loading, Panel, Stat } from './primitives'
import { RulScatter } from './charts'
import { Explainer } from './Explainer'

const CONFOUNDER_PLAIN: Record<string, string> = {
  load_reduction: 'Gets lighter as it runs (fuel, payload)',
  environment_drift: 'Conditions improve for everyone at once',
  interference: 'Other assets get in the way',
  operator_variation: 'Different drivers behave differently',
  operating_mode: 'Runs in distinct modes or duty cycles',
}

export function BeyondRacing() {
  const [data, setData] = useState<CrossIndustryResult | null>(null)
  const [business, setBusiness] = useState<BusinessReport | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([advanced.crossIndustry(), advanced.business()])
      .then(([c, b]) => {
        setData(c)
        setBusiness(b)
      })
      .catch((e) => setError(String(e.message ?? e)))
  }, [])

  if (error) return <ErrorNote error={error} />
  if (!data) return <Loading what="the cross-domain results" />

  const transfer = data.validated_transfer

  return (
    <div className="space-y-3">
      <Explainer id="beyond" question="Why would a tyre model work on anything else?">
        <p>
          Strip the motorsport words away and the problem is not about tyres at
          all: <strong>something wears out while it works, you cannot see its
          condition directly, and the signal you can see is contaminated by
          conditions that also change over time.</strong>
        </p>
        <p>
          That description fits a jet engine, a truck tyre, a battery and a
          bearing. So the estimator was written to consume a description of an
          asset rather than a tyre — what counts as "age", what counts as
          "performance", and which confounders exist. Pointing it at engines
          required no change to the model.
        </p>
      </Explainer>

      {transfer && (
        <Panel title="Proof on a second asset class" aside="real data, real ground truth">
          <div className="flex flex-col gap-5 lg:flex-row">
            <div className="lg:w-[42%]">
              <p className="mb-3 max-w-[52ch] text-[13px] leading-relaxed text-ink">
                Formula 1 cannot prove this. Public F1 data contains no measured
                tyre wear, so there is no true answer to check a degradation
                estimate against.
              </p>
              <p className="mb-3 max-w-[52ch] text-[12.5px] leading-relaxed text-ink-dim">
                <strong className="text-ink">{transfer.dataset}</strong> does.
                It is NASA's turbofan benchmark: engines run to failure, with
                published remaining-life labels and a large body of comparable
                published results. The same estimator was pointed at it with no
                tyre-specific code.
              </p>
              <p className="max-w-[52ch] text-[11.5px] leading-relaxed text-ink-faint">
                {transfer.note}
              </p>
            </div>

            <div className="flex-1">
              <div className="grid grid-cols-2 gap-5 sm:grid-cols-4">
                <Stat
                  label="Engines scored"
                  value={String(transfer.n_engines_scored)}
                />
                <Stat
                  label="Remaining-life error"
                  value={transfer.rul_rmse.toFixed(0)}
                  unit="cycles"
                  tone="warm"
                />
                <Stat
                  label="Sensors used"
                  value={`${transfer.n_sensors_used} of 21`}
                  hint="the rest carry no trend"
                />
                <Stat
                  label="Predicted early"
                  value={`${(transfer.fraction_early * 100).toFixed(0)}%`}
                  hint="the safe direction to err"
                />
              </div>

              <div className="mt-5 border-t border-line pt-4">
                <div className="mb-2 text-[11px] text-ink-faint">
                  What transferred without modification
                </div>
                <ul className="space-y-1 text-[12px] text-ink-dim">
                  <li>The latent state model — a level and a rate that drift over time</li>
                  <li>Pooling a degradation baseline across a fleet of units</li>
                  <li>The uncertainty machinery, unchanged</li>
                  <li>Remaining-life projection to a threshold</li>
                </ul>
                <div className="mt-3 text-[11px] text-ink-faint">What changed</div>
                <ul className="space-y-1 text-[12px] text-ink-dim">
                  <li>"Laps" became "flight cycles"</li>
                  <li>"Lap time" became a health index fused from 12 sensors</li>
                  <li>Fuel and traffic switched off — engines have neither</li>
                </ul>
              </div>
            </div>
          </div>
        </Panel>
      )}

      {transfer?.predictions && transfer.truths && (
        <Panel
          title="Every engine, predicted against actual"
          aside={`${transfer.predictions.length} held-out engines`}
        >
          <div className="grid gap-4 lg:grid-cols-[1.3fr_1fr]">
            <RulScatter predictions={transfer.predictions} truths={transfer.truths} />
            <div className="space-y-3 text-[12.5px] leading-relaxed text-ink-dim">
              <p>
                Each dot is one engine. The dashed diagonal is a perfect
                prediction; distance from it is the error.
              </p>
              <p>
                <strong className="text-good">Green points sit below the line</strong> —
                the model predicted less life than the engine actually had. That is
                the safe direction to be wrong, and NASA&rsquo;s own scoring function
                penalises the other direction far more heavily, because predicting
                an engine has life it does not is what grounds aircraft.
              </p>
              <p>
                The scatter is what an RMSE hides. A single number cannot tell you
                whether the errors are symmetric, and for a prognostics model the
                asymmetry is the safety-relevant property.
              </p>
              <p className="text-[11.5px] text-ink-faint">
                Predictions are capped at 125 cycles, the piecewise-linear
                convention used throughout the C-MAPSS literature — engines show
                essentially no degradation early in life, so an unbounded
                extrapolation from a flat trend is meaningless.
              </p>
            </div>
          </div>
        </Panel>
      )}

      <Panel title="The same engine, three asset classes">
        <div className="grid gap-3 md:grid-cols-3">
          {data.profiles.map((profile) => {
            const validated = profile.asset_type === 'turbofan' || profile.asset_type === 'f1_tyre'
            return (
              <div
                key={profile.asset_type}
                className="border border-line p-3"
                style={{ borderColor: validated ? 'var(--color-line-bright)' : undefined }}
              >
                <div className="mb-1 flex items-baseline justify-between gap-2">
                  <span className="text-[12.5px] font-medium text-ink">
                    {profile.display_name}
                  </span>
                  <span
                    className="shrink-0 text-[9.5px]"
                    style={{
                      color: validated ? 'var(--color-good)' : 'var(--color-medium)',
                    }}
                  >
                    {validated ? 'validated' : 'architecture only'}
                  </span>
                </div>
                <dl className="mb-2 space-y-0.5 text-[11px]">
                  <div className="flex justify-between">
                    <dt className="text-ink-faint">measured against</dt>
                    <dd className="text-ink-dim">{profile.age_unit}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-ink-faint">observed signal</dt>
                    <dd className="text-ink-dim">{profile.performance_unit}</dd>
                  </div>
                </dl>
                <div className="mb-2">
                  <div className="mb-1 text-[10px] text-ink-faint">
                    confounders present
                  </div>
                  <div className="space-y-0.5">
                    {profile.confounders.map((c) => (
                      <div key={c} className="text-[10.5px] text-ink-dim">
                        {CONFOUNDER_PLAIN[c] ?? c}
                      </div>
                    ))}
                  </div>
                </div>
                <p className="text-[10.5px] leading-relaxed text-ink-faint">{profile.notes}</p>
              </div>
            )
          })}
        </div>
      </Panel>

      <div className="grid gap-3 lg:grid-cols-[1fr_1fr]">
        {business && (
          <Panel title="What better estimates are worth" aside="in racing">
            <div className="space-y-3">
              {business.estimates.map((estimate) => (
                <div key={estimate.metric} className="border-t border-line/60 pt-2.5 first:border-0 first:pt-0">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-[12.5px] text-ink">{estimate.metric}</span>
                    <span className="num shrink-0 text-[15px] text-ink">
                      {estimate.value.toFixed(estimate.unit === '%' ? 1 : 2)}
                      <span className="ml-1 text-[10px] text-ink-faint">{estimate.unit}</span>
                    </span>
                  </div>
                  <div className="mt-1 flex items-start gap-2">
                    <ConfidenceTag level={estimate.confidence} />
                    <p className="max-w-[46ch] text-[11px] leading-relaxed text-ink-faint">
                      {estimate.derivation}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        )}

        <Panel
          title="Commercial fleets"
          aside={<span className="text-medium">illustrative arithmetic</span>}
        >
          <div className="mb-4 border-l-2 border-medium bg-medium/[0.05] px-3 py-2.5">
            <p className="max-w-[52ch] text-[12px] leading-relaxed text-ink-dim">
              <strong className="text-medium">This is not a result.</strong> TyreMind
              has no fleet validation, because no public dataset pairs tyre tread
              depth with vehicle telematics. We looked; it does not exist. The
              numbers below show the arithmetic of the opportunity, nothing more.
            </p>
          </div>

          <div className="space-y-3">
            {data.fleet_illustration.estimates.map((estimate) => (
              <div key={estimate.metric}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-[12px] text-ink-dim">{estimate.metric}</span>
                  <span className="num text-[14px] text-ink-dim">
                    {estimate.value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </span>
                </div>
                <p className="mt-0.5 max-w-[46ch] text-[10.5px] leading-relaxed text-ink-faint">
                  {estimate.derivation}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-4 border-t border-line pt-3">
            <p className="max-w-[52ch] text-[11.5px] leading-relaxed text-ink-dim">
              {data.honest_summary}
            </p>
          </div>
        </Panel>
      </div>
    </div>
  )
}

function ConfidenceTag({ level }: { level: string }) {
  const colour =
    level === 'measured'
      ? 'var(--color-good)'
      : level === 'estimated'
        ? 'var(--color-medium)'
        : 'var(--color-ink-faint)'
  return (
    <span
      className="mt-0.5 shrink-0 border px-1.5 py-px text-[9.5px] uppercase"
      style={{ borderColor: colour, color: colour }}
    >
      {level}
    </span>
  )
}
