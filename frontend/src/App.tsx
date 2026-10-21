import { useEffect, useRef, useState } from 'react'
import { fetchMeta, search, type Meta, type SearchResponse, type SearchSettings } from './api'
import { QuestionForm } from './components/QuestionForm'
import { ResultCard } from './components/ResultCard'
import { SearchOverview } from './components/SearchOverview'
import { Sidebar } from './components/Sidebar'

// Placeholder values for the first render only: searching stays disabled until /api/meta has loaded
// and replaced them with the server's defaults.
const INITIAL_SETTINGS: SearchSettings = {
  method: 'hybrid',
  season: '2025-26',
  useEligibility: true,
  statWeight: 0.6,
  missingPolicy: 'zero',
}

type SearchState =
  | { kind: 'idle' }
  | { kind: 'searching' }
  | { kind: 'done'; response: SearchResponse }
  | { kind: 'failed'; message: string }

function statusText(state: SearchState): string {
  if (state.kind === 'searching') return 'Searching. Writing explanations can take up to half a minute.'
  if (state.kind === 'done') {
    const count = state.response.results.length
    return count === 0 ? 'No players matched.' : `Found ${count} player${count === 1 ? '' : 's'}.`
  }
  return ''
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [metaError, setMetaError] = useState<string | null>(null)
  const [settings, setSettings] = useState<SearchSettings>(INITIAL_SETTINGS)
  const [question, setQuestion] = useState('')
  const [state, setState] = useState<SearchState>({ kind: 'idle' })
  const inFlight = useRef<AbortController | null>(null)

  useEffect(() => {
    fetchMeta()
      .then((loaded) => {
        setMeta(loaded)
        setSettings((current) => ({
          ...current,
          season: loaded.default_season,
          statWeight: loaded.defaults.stat_weight,
          missingPolicy: loaded.defaults.missing_semantic_policy,
        }))
      })
      .catch((error: Error) => setMetaError(error.message))
  }, [])

  const runSearch = async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || !meta) return
    inFlight.current?.abort()
    const controller = new AbortController()
    inFlight.current = controller
    setState({ kind: 'searching' })
    try {
      const response = await search(trimmed, settings, controller.signal)
      setState({ kind: 'done', response })
    } catch (error) {
      if ((error as Error).name === 'AbortError') return
      setState({ kind: 'failed', message: (error as Error).message || 'The search failed. Try again.' })
    }
  }

  return (
    <div className="layout">
      <Sidebar meta={meta} settings={settings} onChange={setSettings} />

      <main className="main">
        <header className="masthead">
          <h1 className="title">FullCourt</h1>
          <p className="lede">
            Ask about NBA players' statistics and playing style. Every number comes from the loaded season's data,
            and every style claim links to the profile it came from.
          </p>
        </header>

        {metaError && (
          <p className="notice notice-error" role="alert">
            Search isn't available yet. {metaError} Then reload this page.
          </p>
        )}

        <QuestionForm
          question={question}
          examples={meta?.examples ?? []}
          availability={meta ? 'ready' : metaError ? 'unavailable' : 'loading'}
          searching={state.kind === 'searching'}
          onQuestionChange={setQuestion}
          onSearch={runSearch}
        />

        <p className="status" role="status" aria-live="polite">{statusText(state)}</p>

        {state.kind === 'failed' && <p className="notice notice-error" role="alert">{state.message}</p>}

        {state.kind === 'idle' && !metaError && (
          <p className="empty">
            Describe the kind of player you're looking for, or open the example questions to see what works well.
          </p>
        )}

        {state.kind === 'done' && (
          <>
            <SearchOverview response={state.response} meta={meta} />
            {state.response.results.length === 0 ? (
              <p className="empty">
                No players fit. Try removing a number from the question, switching the method, or turning off the
                rotation-players filter.
              </p>
            ) : (
              <section aria-label="Ranked players">
                <h2 className="visually-hidden">Ranked players</h2>
                <ol className="results">
                  {state.response.results.map((player) => (
                    <li key={player.player_id}>
                      <ResultCard player={player} method={state.response.summary.method} />
                    </li>
                  ))}
                </ol>
              </section>
            )}
          </>
        )}

        <footer className="footer">
          <p>
            Statistics come from Basketball-Reference. Style profiles are written from the same season's statistics
            by a fixed template; they are not scouting reports.
          </p>
        </footer>
      </main>
    </div>
  )
}
