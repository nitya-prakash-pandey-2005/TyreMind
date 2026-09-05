/**
 * The circuit in three dimensions, coloured by what it does to the tyres.
 *
 * This is not decoration. The geometry is the real racing line from position
 * telemetry, and the colour along it is the frictional load the physics layer
 * computes at each point. So the bright sections are literally where the tyre
 * is being worked hardest — Parabolica at Monza, Copse at Silverstone — and a
 * viewer can see *where* on the lap the degradation comes from, which no table
 * conveys.
 *
 * Rendered with drei's `Line` rather than the `<line>` intrinsic. Two reasons,
 * both of which cost an hour to find:
 *
 *   - `<line>` collides with the SVG element of the same name in JSX, so React
 *     renders an empty SVG node and the canvas stays black with no error.
 *   - `LineBasicMaterial.linewidth` is ignored on nearly every platform, a
 *     long-standing WebGL limitation. drei's Line uses a shader-based fat line,
 *     so width actually applies.
 *
 * Performance follows the R3F guidance: `frameloop="demand"` so nothing renders
 * unless something changed, and the point array is built once and memoised.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import { useThemeColours } from '../lib/theme'

export interface TrackGeometry {
  circuit: string
  year: number
  x: number[]
  y: number[]
  z: number[]
  speed_kmh: number[]
  lateral_g: number[]
  longitudinal_g: number[]
  tyre_load: number[]
  throttle: number[]
  brake: number[]
  stats: {
    lap_time_s: number | null
    driver: string
    peak_lateral_g: number
    peak_braking_g: number
    top_speed_kmh: number
    n_points: number
    track_length_m: number
  }
}

export type ColourMode = 'tyre_load' | 'speed_kmh' | 'lateral_g'

export const MODE_LABEL: Record<ColourMode, string> = {
  tyre_load: 'Tyre loading',
  speed_kmh: 'Speed',
  lateral_g: 'Cornering force',
}

export const MODE_HELP: Record<ColourMode, string> = {
  tyre_load:
    'How hard the tyres are working at each point, from the physics layer. Bright means the contact patch is dissipating the most energy — this is where a stint gets used up.',
  speed_kmh: 'Straightforward speed, useful for orienting yourself on the lap.',
  lateral_g:
    'Cornering load, computed from the curvature of the racing line — so it needs no vehicle model, just the geometry the car actually drove.',
}

/** Centre and scale a track so any circuit fits the same viewing box. */
function useTrackPoints(track: TrackGeometry) {
  return useMemo(() => {
    const { x, y, z } = track
    const cx = (Math.max(...x) + Math.min(...x)) / 2
    const cy = (Math.max(...y) + Math.min(...y)) / 2
    const cz = (Math.max(...z) + Math.min(...z)) / 2
    const extent = Math.max(Math.max(...x) - Math.min(...x), Math.max(...y) - Math.min(...y))
    const scale = 100 / (extent || 1)

    const points = x.map(
      (_, i) =>
        new THREE.Vector3(
          (x[i] - cx) * scale,
          // Elevation exaggerated. Circuits vary by tens of metres over
          // kilometres; at true proportions Eau Rouge is a flat line.
          (z[i] - cz) * scale * 6,
          (y[i] - cy) * scale,
        ),
    )
    // Close the loop back to the start line.
    points.push(points[0].clone())
    return points
  }, [track])
}

/** Three-stop ramp: cool where the tyre is idle, hot where it is working. */
function useRampColours(track: TrackGeometry, mode: ColourMode) {
  const colours = useThemeColours()

  return useMemo(() => {
    const raw = track[mode]
    const values = mode === 'lateral_g' ? raw.map(Math.abs) : raw
    const lo = Math.min(...values)
    const hi = Math.max(...values)
    const span = hi - lo || 1

    const cold = new THREE.Color(colours.fuel)
    const warm = new THREE.Color(colours.medium)
    const hot = new THREE.Color(colours.alert)

    const ramp = (t: number) => {
      const c = new THREE.Color()
      if (t < 0.5) c.lerpColors(cold, warm, t * 2)
      else c.lerpColors(warm, hot, (t - 0.5) * 2)
      return c
    }

    const list = values.map((v) => ramp((v - lo) / span))
    list.push(list[0])
    return list
  }, [track, mode, colours])
}

function Scene({
  track,
  mode,
  progress,
}: {
  track: TrackGeometry
  mode: ColourMode
  progress: number
}) {
  const colours = useThemeColours()
  const points = useTrackPoints(track)
  const vertexColors = useRampColours(track, mode)
  const { invalidate } = useThree()

  useEffect(() => invalidate(), [vertexColors, invalidate])

  const carRef = useRef<THREE.Mesh>(null)
  useFrame(() => {
    if (!carRef.current) return
    const i = Math.floor(progress * (points.length - 1))
    const p = points[Math.max(0, Math.min(i, points.length - 1))]
    carRef.current.position.set(p.x, p.y + 2.2, p.z)
  })

  // A faint copy of the line dropped below the surface, so the 3D shape and
  // the elevation changes read against a reference plane.
  const shadowPoints = useMemo(
    () => points.map((p) => new THREE.Vector3(p.x, -14, p.z)),
    [points],
  )

  return (
    <group>
      <Line points={shadowPoints} color={colours.line} lineWidth={1} transparent opacity={0.55} />
      <Line points={points} vertexColors={vertexColors} lineWidth={3.4} />

      {/* Start/finish. */}
      <mesh position={points[0]}>
        <sphereGeometry args={[2.0, 14, 14]} />
        <meshBasicMaterial color={colours.good} />
      </mesh>

      {/* The car working its way round. */}
      <mesh ref={carRef}>
        <sphereGeometry args={[2.6, 18, 18]} />
        <meshBasicMaterial color={colours.ink} />
      </mesh>
    </group>
  )
}

function OrbitCamera({ enabled, progress }: { enabled: boolean; progress: number }) {
  const { camera, invalidate } = useThree()
  const angle = useRef(0.9)

  useFrame((_, delta) => {
    if (enabled) angle.current += delta * 0.15
    const radius = 118
    camera.position.set(Math.cos(angle.current) * radius, 74, Math.sin(angle.current) * radius)
    camera.lookAt(0, 0, 0)
    invalidate()
  })

  // Keep the car moving even when the camera is paused.
  useEffect(() => invalidate(), [progress, invalidate])
  return null
}

export function Circuit3D({
  track,
  mode,
  rotating,
  progress,
}: {
  track: TrackGeometry
  mode: ColourMode
  rotating: boolean
  progress: number
}) {
  const colours = useThemeColours()

  // R3F sizes the canvas from a ResizeObserver on its container. When the view
  // mounts as part of a route switch the observer can miss the first layout,
  // leaving the canvas at its 300x150 HTML default inside a correctly-sized
  // container -- a black rectangle with no error anywhere. Dispatching a resize
  // once after mount forces the measurement to run.
  useEffect(() => {
    const nudge = () => window.dispatchEvent(new Event('resize'))
    const timers = [requestAnimationFrame(nudge), window.setTimeout(nudge, 120)]
    return () => {
      cancelAnimationFrame(timers[0])
      clearTimeout(timers[1])
    }
  }, [])

  return (
    <div className="h-full w-full" style={{ background: colours.ground }}>
      <Canvas
        frameloop="always"
        camera={{ position: [84, 74, 84], fov: 46, near: 1, far: 3000 }}
        dpr={[1, 2]}
        gl={{ antialias: true }}
        resize={{ debounce: 0, scroll: false }}
      >
        <OrbitCamera enabled={rotating} progress={progress} />
        <Scene track={track} mode={mode} progress={progress} />
      </Canvas>
    </div>
  )
}

export function CircuitLegend({ mode }: { mode: ColourMode }) {
  const colours = useThemeColours()
  return (
    <div className="flex items-center gap-3">
      <span className="text-[10px] text-ink-faint">low</span>
      <div
        className="h-1.5 w-32"
        style={{
          background: `linear-gradient(90deg, ${colours.fuel}, ${colours.medium}, ${colours.alert})`,
        }}
      />
      <span className="text-[10px] text-ink-faint">high</span>
      <span className="text-[10.5px] text-ink-dim">{MODE_LABEL[mode]}</span>
    </div>
  )
}

/** Drives the car marker around the lap. */
export function useLapProgress(running: boolean, seconds = 9) {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    if (!running) return
    let frame = 0
    const started = performance.now()
    const tick = () => {
      setProgress((((performance.now() - started) / 1000) % seconds) / seconds)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [running, seconds])

  return progress
}
