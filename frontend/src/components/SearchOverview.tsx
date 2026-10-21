import { memo } from 'react'
import type { Meta, SearchResponse } from '../api'

interface SearchOverviewProps {
  response: SearchResponse
  meta: Meta | null
}

const STAGE_LABELS: Record<string, string> = {
  parse: 'Reading the question',
  sql: 'Statistics lookup',
  vector_search: 'Style search',
  ranking: 'Ranking',
  generation: 'Explanations',
}

export const SearchOverview = memo(function SearchOverview({ response, meta }: SearchOverviewProps) {
  const { summary } = response
  const notices = [...response.messages, ...response.warnings]

  return (
    <section className="overview" aria-labelledby="overview-heading">
      <h2 id="overview-heading" className="visually-hidden">How this search ran</h2>
      <dl className="summary">
        <div>
          <dt>Method</dt>
          <dd>{summary.method_label}</dd>
        </div>
        <div>
          <dt>Season</dt>
          <dd>{summary.season}{summary.season_was_defaulted && ' (default)'}</dd>
        </div>
        <div>
          <dt>Question read by</dt>
          <dd>{summary.parser === 'llm' ? 'LLM' : 'Keyword rules'}</dd>
        </div>
        <div>
          <dt>Candidates after filters</dt>
          <dd>{summary.candidate_count}</dd>
        </div>
      </dl>

      {notices.length > 0 && (
        <ul className="notices" aria-label="Notes about this search">
          {notices.map((notice) => <li key={notice} className="notice">{notice}</li>)}
        </ul>
      )}

      {response.unsupported.length > 0 && (
        <ul className="notices" aria-label="Parts of the question this data can't answer">
          {response.unsupported.map((item) => (
            <li key={item} className="notice notice-warning">Not covered by this data: {item}</li>
          ))}
        </ul>
      )}

      <details className="disclosure">
        <summary>Search plan and timings</summary>
        <div className="plan-grid">
          <div>
            <h3 className="panel-heading">Search plan</h3>
            <pre className="plan-json">{JSON.stringify(response.plan, null, 2)}</pre>
          </div>
          <div>
            <h3 className="panel-heading">Time per stage</h3>
            <table className="data-table">
              <thead>
                <tr><th scope="col">Stage</th><th scope="col" className="numeric">Milliseconds</th></tr>
              </thead>
              <tbody>
                {Object.entries(response.timings_ms).map(([stage, ms]) => (
                  <tr key={stage}>
                    <td>{STAGE_LABELS[stage] ?? stage}</td>
                    <td className="numeric">{ms.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {summary.eligibility_applied && meta && (
              <p className="field-help">
                Only rotation players were considered: at least {meta.defaults.min_games} games and{' '}
                {meta.defaults.min_minutes} minutes per game.
              </p>
            )}
          </div>
        </div>
      </details>
    </section>
  )
})
