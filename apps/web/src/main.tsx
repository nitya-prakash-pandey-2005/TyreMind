import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ThemeProvider, initialTheme } from './lib/theme'

// Applied before React mounts so the first paint is already the right theme.
// Setting it inside an effect produces a visible flash of the wrong one.
document.documentElement.dataset.theme = initialTheme()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </StrictMode>,
)
