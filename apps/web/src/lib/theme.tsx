/**
 * Theme state, and a way for canvas-based charts to react to it.
 *
 * Two things make this less trivial than it looks:
 *
 * 1. **ECharts and three.js draw to canvas**, so they cannot resolve
 *    `var(--colour)`. They need concrete values, re-read whenever the theme
 *    changes. `useThemeColour` exists so a chart re-renders with the right
 *    palette instead of keeping the dark one on a white page.
 * 2. **The choice has to survive a reload**, and the first paint must already be
 *    correct — flashing dark before switching to light is worse than not having
 *    the feature.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

export type Theme = 'dark' | 'light'

const STORAGE_KEY = 'tyremind.theme'

interface ThemeContextValue {
  theme: Theme
  toggle: () => void
  /** Increments on every theme change, so charts can key off it and redraw. */
  revision: number
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: 'dark',
  toggle: () => undefined,
  revision: 0,
})

/** Read the stored preference, falling back to the OS setting. */
export function initialTheme(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'dark' || stored === 'light') return stored
  } catch {
    /* private browsing or storage disabled; fall through to the OS setting */
  }
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() =>
    typeof window === 'undefined' ? 'dark' : initialTheme(),
  )
  const [revision, setRevision] = useState(0)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      /* nothing to recover from; the theme still applies this session */
    }
    // Bump after the attribute lands, so anything re-reading computed styles
    // sees the new values rather than the old ones.
    setRevision((r) => r + 1)
  }, [theme])

  const toggle = useCallback(
    () => setTheme((t) => (t === 'dark' ? 'light' : 'dark')),
    [],
  )

  const value = useMemo(() => ({ theme, toggle, revision }), [theme, toggle, revision])
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext)
}

/**
 * Concrete colours for canvas rendering, refreshed whenever the theme changes.
 *
 * Charts should read every colour from here rather than hard-coding hex values,
 * which is what previously left axis labels invisible on a light background.
 */
export function useThemeColours() {
  const { theme, revision } = useTheme()

  return useMemo(() => {
    const read = (name: string, fallback: string) => {
      if (typeof window === 'undefined') return fallback
      return (
        getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
      )
    }
    return {
      theme,
      ink: read('--color-ink', '#e4eaed'),
      inkDim: read('--color-ink-dim', '#8fa3ae'),
      inkFaint: read('--color-ink-faint', '#5d6f7a'),
      line: read('--color-line', '#26343d'),
      raised: read('--color-raised', '#1d272e'),
      surface: read('--color-surface', '#151d23'),
      ground: read('--color-ground', '#0e1418'),
      alert: read('--color-alert', '#ff8a5b'),
      good: read('--color-good', '#4bbf8a'),
      fuel: read('--color-fuel', '#4fa8c5'),
      track: read('--color-track', '#7b8fa1'),
      traffic: read('--color-traffic', '#b47fd0'),
      residual: read('--color-residual', '#5a6b76'),
      soft: read('--color-soft', '#e8352e'),
      medium: read('--color-medium', '#f5c518'),
      hard: read('--color-hard', '#ededed'),
    }
    // revision is the dependency that matters: it changes after the theme
    // attribute is applied, which is when the computed values become correct.
  }, [theme, revision])
}

/** Compound colour resolved for canvas, theme-aware. */
export function useCompoundColour() {
  const colours = useThemeColours()
  return useCallback(
    (compound: string) => {
      switch (compound?.toUpperCase()) {
        case 'SOFT':
          return colours.soft
        case 'MEDIUM':
          return colours.medium
        case 'HARD':
          return colours.hard
        default:
          return colours.inkDim
      }
    },
    [colours],
  )
}

/**
 * Icon-only theme switch.
 *
 * Drawn as inline SVG rather than a unicode glyph: the sun and moon characters
 * render inconsistently across fonts and platforms, and one of them falls back
 * to an emoji on Windows, which looks nothing like the rest of the interface.
 *
 * The icon shows the theme you will get, not the one you are in -- the
 * prevailing convention, and the one that makes a single unlabelled button
 * unambiguous. The accessible name spells it out either way.
 */
export function ThemeToggle() {
  const { theme, toggle } = useTheme()
  const next = theme === 'dark' ? 'light' : 'dark'

  return (
    <button
      onClick={toggle}
      title={`Switch to ${next} theme`}
      aria-label={`Switch to ${next} theme`}
      className="flex h-6 w-6 items-center justify-center border border-line text-ink-dim transition-colors hover:border-line-bright hover:text-ink"
    >
      {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
    </button>
  )
}

function SunIcon() {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden fill="none"
      stroke="currentColor" strokeWidth="1.3" strokeLinecap="round">
      <circle cx="8" cy="8" r="3" />
      <path d="M8 1.4v1.6M8 13v1.6M1.4 8h1.6M13 8h1.6M3.3 3.3l1.1 1.1M11.6 11.6l1.1 1.1M12.7 3.3l-1.1 1.1M4.4 11.6l-1.1 1.1" />
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden fill="none"
      stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round">
      <path d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8Z" />
    </svg>
  )
}
