import { memo } from 'react'
import type { Method, PlayerResult } from '../api'
import { ScoreBreakdown } from './ScoreBreakdown'
import { StyleEvidence } from './StyleEvidence'

interface ResultCardProps {
  player: PlayerResult
  method: Method
}

function teamText(player: PlayerResult): string {
  if (!player.team) return 'team not listed'
  return player.teams && player.teams !== player.team ? `${player.team} (${player.teams.split(',').join(', ')})` : player.team
}

function ScoreBar({ score }: { score: PlayerResult['score'] }) {
  const stat = Math.max(0, score.stat_contribution)
  const style = Math.max(0, score.style_contribution)
  const description = `Score ${score.final.toFixed(2)} out of 1: ${stat.toFixed(2)} from statistics, ${style.toFixed(2)} from style match.`
  return (
    <div className="score">
      <p className="score-value">
        <span className="score-number">{score.final.toFixed(2)}</span>
        <span className="score-caption">score</span>
      </p>
      <div className="score-bar" role="img" aria-label={description} title={description}>
        <span className="score-bar-stat" style={{ width: `${stat * 100}%` }} />
        <span className="score-bar-style" style={{ width: `${style * 100}%` }} />
      </div>
      <p className="score-legend" aria-hidden="true">
        <span className="legend-stat">stats {stat.toFixed(2)}</span>
        <span className="legend-style">style {style.toFixed(2)}</span>
      </p>
    </div>
  )
}

// Memoized: results don't depend on the sidebar settings, so moving the weight slider shouldn't
// re-render ten cards with their tables.
export const ResultCard = memo(function ResultCard({ player, method }: ResultCardProps) {
  const explanation = player.explanation
  return (
    <article className="result" aria-labelledby={`player-${player.player_id}`}>
      <header className="result-header">
        <p className="result-rank" aria-hidden="true">{player.rank}</p>
        <div className="result-identity">
          <h3 id={`player-${player.player_id}`} className="result-name">
            <span className="visually-hidden">Rank {player.rank}: </span>
            {player.player_name}
          </h3>
          <p className="result-meta">
            {player.position ?? 'Position not listed'} for {teamText(player)}, {player.season}, {player.games} games
          </p>
        </div>
        <ScoreBar score={player.score} />
      </header>

      <dl className="stat-tiles">
        {player.key_stats.map((stat) => (
          <div key={stat.metric} className="stat-tile">
            <dt>{stat.label}</dt>
            <dd>{stat.value}</dd>
          </div>
        ))}
      </dl>

      {explanation && (
        <div className="explanation">
          <h4 className="explanation-label">
            {explanation.source === 'llm' ? 'Explanation, checked against the evidence' : 'Explanation from the template'}
          </h4>
          <p>{explanation.text}</p>
          {explanation.grounding_issues.length > 0 && (
            <p className="field-help">
              The generated explanation was replaced because it {explanation.grounding_issues.join('; ')}.
            </p>
          )}
        </div>
      )}

      <ScoreBreakdown player={player} />
      <StyleEvidence player={player} method={method} />
    </article>
  )
})
