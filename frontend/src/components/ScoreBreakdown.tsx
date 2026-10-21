import type { PlayerResult } from '../api'

const COMPONENT_LABELS: Record<string, string> = {
  statistics: 'statistics',
  semantic: 'style match',
}

function sentenceCase(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1)
}

function formatScore(score: number | null): string {
  return score === null ? 'not available' : score.toFixed(3)
}

export function ScoreBreakdown({ player }: { player: PlayerResult }) {
  const { score, goals } = player
  const used = score.components_used.map((c) => COMPONENT_LABELS[c] ?? c)
  const proxyNotes = [...new Set(goals.map((g) => g.proxy_note).filter((note): note is string => Boolean(note)))]

  return (
    <details className="disclosure">
      <summary>
        <span>Score breakdown<span className="visually-hidden"> for {player.player_name}</span></span>
      </summary>
      <p className="field-help">
        {used.length > 0 ? `This score uses ${used.join(' and ')}.` : 'Nothing in the question could be scored.'}
      </p>
      <table className="data-table">
        <thead>
          <tr><th scope="col">Part</th><th scope="col" className="numeric">Score (0 to 1)</th></tr>
        </thead>
        <tbody>
          <tr><td>Statistical fit</td><td className="numeric">{formatScore(score.stat_fit)}</td></tr>
          <tr>
            <td>Style match{score.style_missing && score.components_used.includes('semantic') && ' (no style profile, default applied)'}</td>
            <td className="numeric">{formatScore(score.style_match)}</td>
          </tr>
        </tbody>
      </table>

      {goals.length > 0 && (
        <div className="table-scroll" role="region" aria-label="What the statistical fit is made of" tabIndex={0}>
        <table className="data-table data-table-wide">
          <caption className="panel-heading">What the statistical fit is made of</caption>
          <thead>
            <tr>
              <th scope="col">Goal</th>
              <th scope="col" className="numeric">Goal score</th>
              <th scope="col">Stat</th>
              <th scope="col" className="numeric">Value</th>
              <th scope="col">Among candidates</th>
            </tr>
          </thead>
          <tbody>
            {goals.flatMap((goal) =>
              goal.metrics.length === 0
                ? [
                    <tr key={goal.label}>
                      <th scope="row">{sentenceCase(goal.label)}</th>
                      <td className="numeric">{formatScore(goal.score)}</td>
                      <td colSpan={3} />
                    </tr>,
                  ]
                : goal.metrics.map((metric, index) => (
                    <tr key={`${goal.label}-${metric.label}`}>
                      {index === 0 && (
                        <>
                          <th scope="row" rowSpan={goal.metrics.length}>{sentenceCase(goal.label)}</th>
                          <td className="numeric" rowSpan={goal.metrics.length}>{formatScore(goal.score)}</td>
                        </>
                      )}
                      <td>{metric.label}</td>
                      <td className="numeric">{metric.value}</td>
                      <td>{metric.pool_rank}</td>
                    </tr>
                  )),
            )}
          </tbody>
        </table>
        </div>
      )}

      {proxyNotes.map((note) => <p key={note} className="field-help">Stand-in stat: {note}</p>)}
      <p className="field-help">
        Statistics from {player.stats_source.name}:{' '}
        {player.stats_source.tables.map((table, index) => (
          <span key={table.url}>
            {index > 0 && ', '}
            <a href={table.url} target="_blank" rel="noreferrer">{table.label}</a>
          </span>
        ))}
        .
      </p>
    </details>
  )
}
