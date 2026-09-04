# Data availability

What public Formula 1 data actually contains, and what it does not. Written
first, because most of the architecture follows from it.

## The matrix

| Variable | Public? | Source | Resolution | Reliability | Used for |
|---|---|---|---|---|---|
| Lap time | Yes | FastF1 timing | Lap | High | The observation |
| Tyre compound | Yes | FastF1 stint data | Stint | High | Pooling degradation |
| Tyre age (`TyreLife`) | Yes | FastF1 | Lap | High | The degradation clock |
| Stint number | Yes | FastF1 | Stint | Medium | Run boundaries |
| Lap start time | Yes | FastF1 | Lap | High | Reconstructing traffic |
| Pit in/out | Yes | FastF1 | Lap | High | Exclusion, pit loss |
| Track status | Yes | FastF1 | Event | Medium | Safety-car exclusion |
| Air / track temperature | Yes | FastF1 weather | ~1 min | Medium | Thermal boundary conditions |
| Speed | Yes | Car telemetry | 4–10 Hz | High | Dynamics |
| Throttle, brake, gear, RPM, DRS | Yes | Car telemetry | 4–10 Hz | High | Driving-style features |
| **X / Y / Z position** | **Yes** | Position telemetry | 4–10 Hz | High | **Curvature, lateral g, per-corner load** |
| Fuel mass | **No** | — | — | — | Confounder, pinned by prior |
| Tyre surface temperature | **No** | — | — | — | Estimated state only |
| Tyre carcass temperature | **No** | — | — | — | Estimated state only |
| Tyre pressure | **No** | — | — | — | Not modelled |
| Slip angle / slip ratio | **No** | — | — | — | **Why Pacejka was cut** |
| Contact patch pressure | **No** | — | — | — | Proxy only |
| **Tread depth / physical wear** | **No** | — | — | — | **Why we estimate performance, not wear** |
| Engine mode / deployment | **No** | — | — | — | Absorbed into noise |
| Setup, ride height, wing level | **No** | — | — | — | Absorbed into run intercept |

## Three consequences

**1. Position telemetry is the most underused public signal.** Differentiating
X/Y twice gives path curvature; curvature with speed gives lateral acceleration
with no vehicle model in between. That single quantity unlocks the g-g trace,
load transfer, and a per-corner picture of tyre loading. Most public F1 analysis
ignores it. `exp06_circuit_asymmetry` shows it is good enough to recover which way
a circuit runs on 7 of 8 circuits, having never been told.

**2. Missing fuel mass is not fatal, but missing fuel *rate* would be.** The
unknown starting mass shifts a run intercept, which a run-level effect absorbs.
Only the burn *slope* matters, and it is collinear with degradation — hence the
physical prior.

**3. No tread depth means no real-world ground truth for wear.** The single most
consequential line in the table. It is why:

- the product estimates *latent performance state* rather than physical wear;
- the ground-truth benchmark is synthetic, and labelled as such;
- the cross-domain validation uses NASA C-MAPSS, which does have run-to-failure
  ground truth;
- fleet claims are labelled architecture rather than evidence.

## What we looked for and did not find

**A public dataset pairing tyre tread depth with vehicle telematics does not
exist.** Searched across fleet-telematics vendors, automotive open datasets
(Waymo, nuScenes, KITTI, comma2k19, A2D2) and the tyre-wear literature. Tread
depth lives in proprietary fleet-management systems. The JRC reports 1.0–1.2 mm
of front tread lost per 10,000 km for passenger cars, which sets a scale but is
not a dataset.

This is why the fleet application is presented as architecture, and why the
cross-domain proof runs on turbofans instead.

## A data-integrity trap worth knowing about

FastF1 matches event names by string similarity and **silently substitutes its
best guess**, logging only a warning. Asking for `"Interlagos"` returns the
**Dutch Grand Prix at Zandvoort** — a different continent, a different circuit,
no error raised.

It was caught only because two circuits produced byte-identical per-corner energy
shares. `data/f1_loader.py` now carries a curated alias table and refuses any
resolved event unrelated to the request.
