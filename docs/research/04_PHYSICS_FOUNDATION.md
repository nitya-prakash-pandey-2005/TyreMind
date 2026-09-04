# Physics foundation

Every function in `src/tyremind/physics/` states which side of a line it sits on.

**Recovered from data, no vehicle model:** speed, longitudinal acceleration, path
curvature, lateral acceleration, distance.

**Model-derived,** and therefore only as good as the parameters in
`configs/physics.yaml`: normal load per corner, frictional power, energy
exposure, thermal state.

**Not available at any price:** slip angle, slip ratio, contact patch pressure,
tyre temperature, tread depth. These need sensors public telemetry does not
carry, and nothing pretends otherwise. Functions returning a proxy say `_proxy`
in the name.

---

## Vehicle dynamics

### Curvature — the quantity that unlocks the rest

Position telemetry gives X/Y at 4–10 Hz. The parametric curvature formula

```
kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
```

is signed: positive is a left-hand turn, negative a right-hand one. That sign is
what makes left/right tyre asymmetry recoverable at all.

Second derivatives amplify noise brutally, so coordinates are smoothed over seven
samples first — roughly one to two seconds, which preserves corners while killing
jitter. It is a real trade-off: too short and straights sprout phantom corners,
too long and genuine direction changes flatten into one.

### Lateral acceleration

```
a_y = v^2 * kappa
```

Straight from kinematics. No vehicle model, no tyre model, no assumption about
grip — the most physically trustworthy derived quantity available from public
data. Clipped at 7 g and zeroed below 8 m/s, because positional glitches
otherwise produce impossible loads that propagate into every downstream figure.

### Aerodynamic load

```
F_down = 0.5 * rho * ClA * v^2
```

ClA is a literature estimate (see `configs/physics.yaml`) and varies with a
setup that is not public. It matters anyway: at 300 km/h an F1 car makes several
times its own weight in downforce, so ignoring it would understate loads in
exactly the high-speed corners that do the most damage.

### Load transfer

Quasi-static rigid-body:

```
longitudinal transfer = m * a_x * h / wheelbase      (braking loads the front)
lateral transfer      = m * a_y * h / track_width    (cornering loads the outside)
```

Ignores suspension compliance, anti-roll distribution, aerodynamic balance shift
with ride height, and the transient during which load actually migrates. Those
matter for setup work; for ranking which corner of the car worked hardest over a
lap, the quasi-static split captures the dominant effect.

### Frictional power — explicitly a proxy

True frictional power is `mu * F_normal * v_slip`, and slip velocity is not
observable without wheel-speed sensors. Slip is approximated as proportional to
demanded acceleration: a tyre asked for more grip is slipping more.

**This is the weakest link in the physics chain.** What survives it is the
*relative* ordering — which corner worked hardest, which lap was more demanding,
which circuit puts more through the rubber — because the unknown constant of
proportionality divides out of every comparison the platform actually makes.
Absolute values in watts are meaningless and are never reported.

---

## Thermal model

Two states, because the distinction is the whole point: a driver can drop surface
temperature in one cool-down lap, but an overheated carcass stays overheated, and
that is what ends stints.

```
C_s dTs/dt = Q_friction - k_sb(Ts - Tb) - h(v)(Ts - T_air) - k_road(Ts - T_track)
C_b dTb/dt = k_sb(Ts - Tb) - h_b(Tb - T_air)
```

Explicit Euler at telemetry sampling rates is comfortably stable — the stiffest
mode has a time constant of several seconds against a step of 0.1–0.25 s. The
step is clamped anyway, because telemetry contains gaps and a two-second hole
would otherwise integrate a heat spike.

**These are estimated states, not measurements.** The coefficients are calibrated
so their output correlates with observed degradation, not against any temperature
sensor — there is none. Absolute degrees are not trustworthy; the relative
structure is what the degradation model consumes.

---

## Wear

### Why not Archard

```
V = K * F_N * L / H
```

Archard is the standard starting point and is retained as a documented baseline
for the ablation. It is used because it is simple, not because it is right for
rubber: a constant coefficient, blind to viscoelasticity, roughness and
temperature. The [rubber wear review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12915245/)
says as much explicitly.

### The energy formulation

```
dW/dt = K(T, compound) * P_friction
```

Rubber wear tracks dissipated frictional energy, and the coefficient is where
temperature enters.

### Temperature: the bidirectional detail that matters

The wear multiplier rises **on both sides** of the working window:

- **Too cold.** Stiff rubber cannot generate grip through hysteresis, so the
  driver slides it, which tears the surface. This is graining.
- **Too hot.** Rubber softens past its useful range and abrades and blisters.
  This is thermal degradation.

An Arrhenius form is the textbook choice and is **wrong here**, because it is
monotone in temperature and so cannot represent the cold side at all — half of
real tyre behaviour. A symmetric quadratic penalty around the window is the
simplest form with the right shape, and public data cannot justify anything more
elaborate. `tests/physics/test_thermal_wear.py` guards this specifically.

---

## What was cut, and why

**Full Pacejka Magic Formula identification.** The Magic Formula needs slip angle
and slip ratio as inputs. Neither is observable from public telemetry. Fitting a
production-grade parameter set would have consumed the build and produced
parameters no available data could constrain. A physically motivated feature
layer that *is* identifiable was built instead.

---

## Independent validation

The whole chain — GPS to curvature to lateral g to per-corner load — is checked
against published fact that never enters the code.

A clockwise circuit is mostly right-hand corners, and cornering right throws load
onto the *left* tyres. So a clockwise circuit must show a left-side energy share
above 50%.

**Result: 7 of 8 circuits recovered correctly.**

| Circuit | Published | Left-turn energy | Left-side load | Correct |
|---|---|---:|---:|:-:|
| Monza | clockwise | 21.5% | 54.9% | yes |
| Barcelona | clockwise | 25.9% | 58.2% | yes |
| Zandvoort | clockwise | 29.5% | 58.9% | yes |
| Silverstone | clockwise | 35.4% | 54.5% | yes |
| Austin | anti-clockwise | 46.2% | 51.1% | **no** |
| Imola | anti-clockwise | 60.1% | 48.1% | yes |
| Singapore | anti-clockwise | 60.9% | 47.3% | yes |
| Interlagos | anti-clockwise | 74.6% | 45.5% | yes |

Austin misses, and is reported as a miss. At 46.2% it is genuinely borderline —
COTA's esses alternate direction, so the circuit really is close to balanced.

Recovering this requires curvature sign, load transfer and energy integration to
all be correct simultaneously. It is the strongest available check that the
physics layer computes what it claims to.

---

## A hypothesis that failed

**Predicted:** cumulative tyre energy would be a better degradation clock than
lap count, because a tyre responds to energy through the contact patch rather
than to distance.

**Result: no meaningful difference.** Mean R² gain −0.0005 over six stints.

**Why:** per-lap energy varies by only **1.9%** within a stint. F1 drivers are
extremely consistent, so cumulative energy is very nearly proportional to lap
count and no improvement is *possible* within a stint.

The cross-session version — whether an energy clock would absorb the systematic
+0.047 s/lap practice-to-race bias, which is plausibly a fuel-load effect —
remains untested and is the obvious next experiment.
