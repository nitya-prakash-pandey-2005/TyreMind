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

export function ThemeToggle() {
  const { theme, toggle } = useTheme()
  const next = theme === 'dark' ? 'light' : 'dark'

  return (
    <button
      onClick={toggle}
      title={`Switch to ${next} theme`}
      aria-label={`Switch to ${next} theme`}
      className="flex items-center gap-1.5 border border-line px-2 py-1 text-[11px] text-ink-dim transition-colors hover:border-line-bright hover:text-ink"
    >
      <span aria-hidden className="text-[12px] leading-none">
        {theme === 'dark' ? '◐' : '◑'}
      </span>
      {theme === 'dark' ? 'Dark' : 'Light'}
    </button>
  )
}
