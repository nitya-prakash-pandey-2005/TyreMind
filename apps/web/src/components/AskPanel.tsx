/**
 * Ask the project about itself.
 *
 * The honest answer to "why should I trust this?" is spread across a research
 * audit, a model card, a limitations register and seven experiment result files.
 * Nobody reads all of that. This makes it answerable in one line.
 *
 * **It retrieves; it does not generate.** Every panel here is a passage lifted
 * verbatim from a document or an experiment output, with its source attached.
 * A generated summary would be smoother and would remove the one property that
 * makes this useful — that every claim can be checked against the file it came
 * from.
 */

import { useCallback, useEffect, useState } from 'react'
import { Panel } from './primitives'
import { Explainer } from './Explainer'

interface AskResult {
  query: string
  results: {
    text: string
    source: string
    heading: string
    kind: string
    citation: string
    score: number
    lexical_rank: number | null
    semantic_rank: number | null
  }[]
  corpus: {
    n_passages: number
    n_sources: number
    by_kind: Record<string, number>
    embedding_backend: string
    retrieval: string
  }
  note: string
}

/** Questions that show the retrieval working on both of its halves. */
const SUGGESTED = [
  'What does this not measure?',
  'How accurate is it against a known answer?',
  'What happens when it rains?',
  'CRPS',
  'Does it work on jet engines?',
  'Which assumptions carry the result?',
  'Where does a simpler model beat you?',
]

export function AskPanel() {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<AskResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const run = useCallback((q: string) => {
    if (!q.trim()) return
    setBusy(true)
    setError('')
    fetch(`/api/ask?q=${encodeURIComponent(q)}&k=5`)
      .then(async (r) => {
        if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText)
        return r.json()
      })
      .then(setResult)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false))
  }, [])

  useEffect(() => {
    run(SUGGESTED[0])
    setQuery(SUGGESTED[0])
  }, [run])

  return (
    <div className="space-y-3">
      <Explainer id="ask" question="What is this?">
        <p>
          A search over TyreMind&rsquo;s own research audit, model card, limitations
          register and every recorded experiment result — so a question about the
          method has an answer with a source attached.
        </p>
        <p>
          <strong>It retrieves rather than generates.</strong> Everything below is
          lifted verbatim from a file in this repository. A generated summary would
          read better and would lose the one property that matters here: you can
          check it.
        </p>
      </Explainer>

      <Panel
        title="Ask about the method"
        aside={
          result
            ? `${result.corpus.n_passages} passages from ${result.corpus.n_sources} files`
            : undefined
        }
      >
        <form
          onSubmit={(e) => {
            e.preventDefault()
            run(query)
          }}
          className="flex gap-2"
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. how do you know the degradation is real?"
            className="min-w-0 flex-1 border border-line bg-raised px-3 py-2 text-[13px] text-ink placeholder:text-ink-faint focus:border-alert focus:outline-none"
          />
          <button
            type="submit"
            disabled={busy}
            className="shrink-0 border border-alert px-4 py-2 text-[12px] text-alert transition-colors hover:bg-alert/10 disabled:opacity-50"
          >
            {busy ? 'Searching…' : 'Search'}
          </button>
        </form>

        <div className="mt-3 flex flex-wrap gap-1.5">
          {SUGGESTED.map((s) => (
            <button
              key={s}
              onClick={() => {
                setQuery(s)
                run(s)
              }}
              className="border border-line px-2 py-1 text-[11px] text-ink-dim transition-colors hover:border-line-bright hover:text-ink"
            >
              {s}
            </button>
          ))}
        </div>

        {error && <div className="mt-3 text-[12px] text-alert">{error}</div>}
      </Panel>

      {result && (
        <Panel
          title={`Passages matching “${result.query}”`}
          aside={`${result.results.length} results`}
        >
          {result.results.length === 0 ? (
            <p className="py-6 text-center text-[12.5px] text-ink-faint">
              Nothing in the documentation matches that closely enough to show. The
              panel returns nothing rather than the least-bad passage — a confident
              citation of an irrelevant paragraph is worse than an empty result.
            </p>
          ) : (
            <div className="space-y-4">
              {result.results.map((hit, i) => (
                <article key={i} className="border-l-2 border-line pl-3.5">
                  <div className="mb-1 flex flex-wrap items-baseline gap-2">
                    <span
                      className="text-[10px]"
                      style={{
                        color:
                          hit.kind === 'result'
                            ? 'var(--color-good)'
                            : 'var(--color-ink-faint)',
                      }}
                    >
                      {hit.kind === 'result' ? 'measured result' : 'documentation'}
                    </span>
                    <span className="num text-[10.5px] text-ink-faint">{hit.source}</span>
                    {hit.heading && (
                      <span className="text-[10.5px] text-ink-dim">— {hit.heading}</span>
                    )}
                  </div>
                  <p className="max-w-[86ch] text-[12.5px] leading-relaxed text-ink">
                    {hit.text}
                  </p>
                  <div className="mt-1 flex gap-3 text-[10px] text-ink-faint">
                    <span>
                      keyword rank {hit.lexical_rank != null ? hit.lexical_rank + 1 : '—'}
                    </span>
                    <span>
                      meaning rank {hit.semantic_rank != null ? hit.semantic_rank + 1 : '—'}
                    </span>
                  </div>
                </article>
              ))}
            </div>
          )}

          <div className="mt-5 border-t border-line pt-3">
            <div className="mb-1 text-[11px] text-ink-faint">How the search works</div>
            <p className="max-w-[80ch] text-[11.5px] leading-relaxed text-ink-dim">
              Two retrievers with opposite weaknesses run in parallel and their
              rankings are merged. <strong>Keyword search</strong> finds exact terms
              like &ldquo;CRPS&rdquo; or a specific number, and misses paraphrase
              entirely. <strong>Meaning search</strong> finds a passage about
              calibration when you ask &ldquo;how sure is it&rdquo;, and is weak on rare
              exact tokens. The two ranks above show which one found each passage.
            </p>
            <p className="mt-1.5 max-w-[80ch] text-[11px] leading-relaxed text-ink-faint">
              {result.corpus.retrieval}. Embeddings: {result.corpus.embedding_backend}
            </p>
          </div>
        </Panel>
      )}
    </div>
  )
}
