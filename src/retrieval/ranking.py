"""Transparent, testable ranking functions.

Normalization
-------------
* Statistics: percentile rank within the candidate pool (after hard filters and eligibility), using
  average ranks for ties: (rank - 1) / (n - 1), in [0, 1]. Metrics where lower is better (turnover
  rate) are inverted. Percentiles make metrics on different scales comparable and blunt outliers.
* Goal score: weighted mean of its metric percentiles (weights in schemas.GOALS sum to 1). If a
  metric is NULL for a player the remaining metric weights are renormalized; if all are NULL the
  goal score is missing. Direction "low" uses 1 - s; "moderate" uses 1 - 2|s - 0.5|.
* Statistical fit: unweighted mean of the available goal scores.
* Semantic relevance: the player's best chunk cosine similarity, min-max scaled across candidates
  that have any description. Players with no description get 0 ("zero" policy) or the median
  scaled score of players who have one ("neutral" policy).
* Final score: stat_weight * stat_fit + semantic_weight * semantic. When one component is
  unavailable for the whole query (no statistical goals, or no descriptions), the other carries
  full weight and the breakdown says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Literal

from src.schemas import GOALS, METRICS, Preference

Direction = Literal["high", "low", "moderate"]

# Metrics used to compare statistical profiles for "players similar to X" queries.
PROFILE_METRICS = [
    "points_per_game",
    "usage_rate",
    "true_shooting_pct",
    "assist_rate",
    "turnover_rate",
    "three_point_attempts_per_game",
    "rebound_rate",
    "steal_rate",
    "block_rate",
]


@dataclass
class MetricEvidence:
    metric: str
    value: float | None
    percentile: float | None


@dataclass
class GoalBreakdown:
    goal: str
    direction: Direction
    score: float | None
    metrics: list[MetricEvidence]
    proxy_note: str | None = None


@dataclass
class ScoreBreakdown:
    player_id: str
    final_score: float
    stat_fit: float | None
    semantic_score: float | None  # scaled to [0, 1] within the pool, after the missing-evidence policy
    semantic_similarity: float | None  # raw cosine similarity of the best chunk; None if no description
    semantic_missing: bool
    goals: list[GoalBreakdown] = field(default_factory=list)
    components_used: list[str] = field(default_factory=list)


def percentile_ranks(values: list[float | None]) -> list[float | None]:
    present = sorted((v, i) for i, v in enumerate(values) if v is not None)
    result: list[float | None] = [None] * len(values)
    n = len(present)
    if n == 1:
        result[present[0][1]] = 0.5
        return result
    i = 0
    while i < n:
        j = i
        while j + 1 < n and present[j + 1][0] == present[i][0]:
            j += 1
        average_rank = (i + j) / 2  # zero-based
        for k in range(i, j + 1):
            result[present[k][1]] = average_rank / (n - 1)
        i = j + 1
    return result


def apply_direction(score: float, direction: Direction) -> float:
    if direction == "low":
        return 1.0 - score
    if direction == "moderate":
        return 1.0 - 2.0 * abs(score - 0.5)
    return score


def metric_percentiles(candidates: list[dict], metric_names: set[str]) -> dict[str, list[float | None]]:
    """Percentile per metric, oriented so that 1.0 is 'better' (turnover rate is inverted)."""
    out = {}
    for name in metric_names:
        metric = METRICS[name]
        ranks = percentile_ranks([c.get(metric.column) for c in candidates])
        out[name] = ranks if metric.higher_is_better else [None if r is None else 1.0 - r for r in ranks]
    return out


def goal_scores(candidates: list[dict], preferences: list[Preference]) -> list[list[GoalBreakdown]]:
    needed = {m for p in preferences for m in GOALS[p.goal].metrics}
    pct = metric_percentiles(candidates, needed)
    per_player: list[list[GoalBreakdown]] = []
    for i, candidate in enumerate(candidates):
        breakdowns = []
        for pref in preferences:
            definition = GOALS[pref.goal]
            evidence, weighted, total_weight = [], 0.0, 0.0
            for name, weight in definition.metrics.items():
                p = pct[name][i]
                evidence.append(MetricEvidence(name, candidate.get(METRICS[name].column), p))
                if p is not None:
                    weighted += weight * p
                    total_weight += weight
            score = apply_direction(weighted / total_weight, pref.direction) if total_weight else None
            breakdowns.append(GoalBreakdown(pref.goal.value, pref.direction, score, evidence, definition.proxy_note))
        per_player.append(breakdowns)
    return per_player


def metric_goal_scores(candidates: list[dict], directions: dict[str, Direction]) -> list[list[GoalBreakdown]]:
    """One single-metric goal per metric, used when the user gave filters but no ranking preference."""
    pct = metric_percentiles(candidates, set(directions))
    return [
        [
            GoalBreakdown(
                f"metric:{name}",
                direction,
                None if pct[name][i] is None else apply_direction(pct[name][i], direction),
                [MetricEvidence(name, candidate.get(METRICS[name].column), pct[name][i])],
            )
            for name, direction in directions.items()
        ]
        for i, candidate in enumerate(candidates)
    ]


def profile_similarity(candidates: list[dict], reference: dict) -> list[float | None]:
    """1 - mean absolute percentile difference from the reference player over PROFILE_METRICS.

    Percentiles are computed over the candidate pool plus the reference player.
    """
    pool = candidates + [reference]
    pct = metric_percentiles(pool, set(PROFILE_METRICS))
    ref_index = len(pool) - 1
    out: list[float | None] = []
    for i in range(len(candidates)):
        diffs = [abs(pct[m][i] - pct[m][ref_index]) for m in PROFILE_METRICS if pct[m][i] is not None and pct[m][ref_index] is not None]
        out.append(1.0 - sum(diffs) / len(diffs) if diffs else None)
    return out


def scale_semantic(similarities: list[float | None], missing_policy: Literal["zero", "neutral"]) -> list[float | None]:
    present = [s for s in similarities if s is not None]
    if not present:
        return [None] * len(similarities)
    low, high = min(present), max(present)
    scaled = [None if s is None else (0.5 if high == low else (s - low) / (high - low)) for s in similarities]
    fill = 0.0 if missing_policy == "zero" else median([s for s in scaled if s is not None])
    return [fill if s is None else s for s in scaled]


def combine(
    player_ids: list[str],
    stat_fits: list[float | None],
    semantic_similarities: list[float | None],
    goals: list[list[GoalBreakdown]] | None,
    stat_weight: float,
    semantic_weight: float,
    missing_policy: Literal["zero", "neutral"] = "zero",
) -> list[ScoreBreakdown]:
    """Combine statistical fit and semantic relevance into a ranked list of breakdowns (best first)."""
    semantic_scaled = scale_semantic(semantic_similarities, missing_policy)
    has_stats = any(s is not None for s in stat_fits)
    has_semantic = any(s is not None for s in semantic_similarities)

    if has_stats and has_semantic:
        ws, wm, used = stat_weight, semantic_weight, ["statistics", "semantic"]
    elif has_stats:
        ws, wm, used = 1.0, 0.0, ["statistics"]
    elif has_semantic:
        ws, wm, used = 0.0, 1.0, ["semantic"]
    else:
        ws, wm, used = 0.0, 0.0, []

    total = ws + wm or 1.0
    results = []
    for i, pid in enumerate(player_ids):
        stat = stat_fits[i]
        sem = semantic_scaled[i]
        final = ((ws * (stat or 0.0)) + (wm * (sem or 0.0))) / total
        results.append(
            ScoreBreakdown(
                player_id=pid,
                final_score=final,
                stat_fit=stat,
                semantic_score=sem,
                semantic_similarity=semantic_similarities[i],
                semantic_missing=semantic_similarities[i] is None,
                goals=goals[i] if goals else [],
                components_used=used,
            )
        )
    # Stable tie-break on player_id keeps rankings deterministic.
    results.sort(key=lambda r: (-r.final_score, r.player_id))
    return results


def mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def statistical_fit(goal_breakdowns: list[GoalBreakdown]) -> float | None:
    return mean_or_none([g.score for g in goal_breakdowns])

