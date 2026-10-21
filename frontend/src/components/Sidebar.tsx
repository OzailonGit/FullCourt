import { useId, useState } from 'react'
import type { Meta, MissingPolicy, SearchSettings } from '../api'

interface SidebarProps {
  meta: Meta | null
  settings: SearchSettings
  onChange: (settings: SearchSettings) => void
}

const MISSING_POLICY_LABELS: Record<MissingPolicy, string> = {
  zero: 'No style credit',
  neutral: 'Median style score',
}

export function Sidebar({ meta, settings, onChange }: SidebarProps) {
  const [open, setOpen] = useState(false)
  const panelId = useId()
  const update = (patch: Partial<SearchSettings>) => onChange({ ...settings, ...patch })
  const defaults = meta?.defaults
  const isHybrid = settings.method === 'hybrid'

  return (
    <aside className="sidebar" aria-label="Search settings">
      <button
        type="button"
        className="sidebar-toggle"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen(!open)}
      >
        Search settings
      </button>

      <div id={panelId} className="sidebar-panel" data-open={open}>
        <h2 className="sidebar-heading">Search settings</h2>

        <fieldset className="field">
          <legend className="field-label">Retrieval method</legend>
          {(meta?.methods ?? []).map((method) => (
            <label key={method.value} className="radio">
              <input
                type="radio"
                name="method"
                value={method.value}
                checked={settings.method === method.value}
                onChange={() => update({ method: method.value })}
              />
              <span>{method.label}</span>
            </label>
          ))}
        </fieldset>

        <div className="field">
          <label className="field-label" htmlFor="season">Season</label>
          <select
            id="season"
            value={settings.season}
            onChange={(event) => update({ season: event.target.value })}
            aria-describedby="season-help"
          >
            {(meta?.seasons ?? [settings.season]).map((season) => (
              <option key={season} value={season}>{season}</option>
            ))}
          </select>
          <p id="season-help" className="field-help">Used when the question doesn't name a season.</p>
        </div>

        <div className="field">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={settings.useEligibility}
              onChange={(event) => update({ useEligibility: event.target.checked })}
              aria-describedby="eligibility-help"
            />
            <span>
              Only rotation players
              {defaults && ` (at least ${defaults.min_games} games and ${defaults.min_minutes} minutes per game)`}
            </span>
          </label>
          <p id="eligibility-help" className="field-help">
            Keeps a player with two games from topping a rate stat. It's a default, not something read from your
            question, and the style-profiles-only method ignores it.
          </p>
        </div>

        <div className="field" data-disabled={!isHybrid}>
          <label className="field-label" htmlFor="stat-weight">
            Statistics weight <output htmlFor="stat-weight">{settings.statWeight.toFixed(2)}</output>
          </label>
          <input
            id="stat-weight"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={settings.statWeight}
            disabled={!isHybrid}
            onChange={(event) => update({ statWeight: Number(event.target.value) })}
            aria-describedby="stat-weight-help"
          />
          <p id="stat-weight-help" className="field-help">
            {isHybrid
              ? `The rest (${(1 - settings.statWeight).toFixed(2)}) goes to style match.`
              : 'Only used by the hybrid method.'}
          </p>
        </div>

        <div className="field">
          <label className="field-label" htmlFor="missing-policy">Players without a style profile</label>
          <select
            id="missing-policy"
            value={settings.missingPolicy}
            onChange={(event) => update({ missingPolicy: event.target.value as MissingPolicy })}
          >
            {(Object.keys(MISSING_POLICY_LABELS) as MissingPolicy[]).map((policy) => (
              <option key={policy} value={policy}>{MISSING_POLICY_LABELS[policy]}</option>
            ))}
          </select>
        </div>

        <div className="sidebar-status">
          <p>{meta ? meta.llm.status : 'Connecting to the API…'}</p>
          {meta && !meta.llm.available && <p>Questions are read by keyword rules and explanations use a template.</p>}
          {meta && (
            <p>
              {meta.index_size > 0
                ? `${meta.index_size} style profiles indexed.`
                : 'No style index found, so style search is off.'}
            </p>
          )}
        </div>
      </div>
    </aside>
  )
}
