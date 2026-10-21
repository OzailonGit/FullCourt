import type { Method, PlayerResult } from '../api'

// Profiles are written as labelled sentences ("Role: …", "Shot diet: …"); one per line is far
// easier to scan than a single paragraph.
function sentences(text: string): string[] {
  return text.split(/(?<=\.)\s+(?=[A-Z])/)
}

interface StyleEvidenceProps {
  player: PlayerResult
  method: Method
}

export function StyleEvidence({ player, method }: StyleEvidenceProps) {
  const { passages, score } = player
  const count = passages.length === 1 ? '1 profile' : `${passages.length} profiles`
  const styleMatch = score.style_match === null ? null : score.style_match.toFixed(2)

  return (
    <details className="disclosure">
      <summary>
        <span>Style evidence<span className="visually-hidden"> for {player.player_name}</span> ({count})</span>
      </summary>
      {method === 'stats' && <p className="field-help">The statistics-only method doesn't use style profiles.</p>}
      {method !== 'stats' && passages.length === 0 && (
        <p className="field-help">There's no style profile for this player.</p>
      )}
      {passages.map((passage) => (
        <figure key={passage.document_id} className="passage">
          <blockquote>
            {sentences(passage.text).map((sentence, index) => <p key={index}>{sentence}</p>)}
          </blockquote>
          <figcaption className="field-help">
            {passage.is_stat_profile ? (
              <>Written from {player.season} Basketball-Reference numbers by a fixed template, not scouting text.</>
            ) : (
              <>
                From <a href={passage.source_url} target="_blank" rel="noreferrer">{passage.source_name}</a>
                {passage.source_section && `, ${passage.source_section}`}
                {passage.published_at && `, revision of ${passage.published_at.slice(0, 10)}`}.
              </>
            )}
            {styleMatch && <> Style match {styleMatch} on a scale from 0 to 1 across these candidates.</>}
          </figcaption>
        </figure>
      ))}
    </details>
  )
}
