// Types mirror the JSON returned by src/api.py. All numbers meant for display arrive already
// formatted by the backend, so the UI never formats a stat itself.

export type Method = 'hybrid' | 'stats' | 'semantic'
export type MissingPolicy = 'zero' | 'neutral'

export interface Meta {
  seasons: string[]
  default_season: string
  methods: { value: Method; label: string }[]
  examples: string[]
  defaults: {
    min_games: number
    min_minutes: number
    stat_weight: number
    missing_semantic_policy: MissingPolicy
  }
  llm: { available: boolean; status: string }
  index_size: number
}

export interface SearchSettings {
  method: Method
  season: string
  useEligibility: boolean
  statWeight: number
  missingPolicy: MissingPolicy
}

export interface KeyStat {
  metric: string
  label: string
  value: string
}

export interface GoalMetric {
  label: string
  value: string
  pool_rank: string
}

export interface Goal {
  label: string
  score: number | null
  proxy_note: string | null
  metrics: GoalMetric[]
}

export interface Passage {
  document_id: string
  text: string
  source_name: string
  source_url: string
  source_section: string | null
  description_type: string
  is_stat_profile: boolean
  license: string | null
  published_at: string | null
}

export interface PlayerResult {
  rank: number
  player_id: string
  player_name: string
  position: string | null
  team: string | null
  teams: string | null
  season: string
  games: string
  key_stats: KeyStat[]
  score: {
    final: number
    stat_fit: number | null
    style_match: number | null
    style_missing: boolean
    components_used: string[]
    stat_contribution: number
    style_contribution: number
  }
  goals: Goal[]
  explanation: { text: string; source: 'llm' | 'template'; grounding_issues: string[] } | null
  passages: Passage[]
  stats_source: { name: string; tables: { label: string; url: string }[] }
}

export interface SearchResponse {
  summary: {
    method: Method
    method_label: string
    season: string
    season_was_defaulted: boolean
    parser: 'llm' | 'rules'
    candidate_count: number
    eligibility_applied: boolean
  }
  plan: Record<string, unknown>
  messages: string[]
  warnings: string[]
  unsupported: string[]
  timings_ms: Record<string, number>
  results: PlayerResult[]
}

const UNREACHABLE =
  "Couldn't reach the FullCourt API. Check that it's running (uvicorn src.api:app) and try again."

// Three different failures need three different fixes: the API isn't reachable at all, the API
// rejected the request (it explains why in `detail`), or the API failed while handling it.
async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) return 'The search settings were not accepted. Check the sidebar values and try again.'
  } catch {
    // The body isn't JSON (a proxy or crash page), so fall through to a message based on the status.
  }
  if (response.status >= 500) {
    return `The search failed on the server (error ${response.status}). Try again; if it keeps happening, check the API's log.`
  }
  return `The API couldn't handle this request (error ${response.status}). Reload the page and try again.`
}

async function request(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init)
  } catch (error) {
    if ((error as Error).name === 'AbortError') throw error
    throw new Error(UNREACHABLE)
  }
}

export async function fetchMeta(): Promise<Meta> {
  const response = await request('/api/meta')
  if (!response.ok) throw new Error(await readError(response))
  return response.json()
}

export async function search(question: string, settings: SearchSettings, signal?: AbortSignal): Promise<SearchResponse> {
  const response = await request('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({
      question,
      method: settings.method,
      season: settings.season,
      use_eligibility: settings.useEligibility,
      stat_weight: settings.statWeight,
      missing_semantic_policy: settings.missingPolicy,
    }),
  })
  if (!response.ok) throw new Error(await readError(response))
  return response.json()
}
