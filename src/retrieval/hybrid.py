"""Run a SearchPlan against the database and the semantic index with one of three methods.

* "stats":    SQL filters + statistical ranking only.
* "semantic": vector retrieval over descriptions only. Ignores SQL filters by design, so it behaves
              like a plain vector-search baseline; candidates are players in the season with a description.
* "hybrid":   SQL filters first, then a weighted combination of statistical fit and semantic relevance.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Literal

from src.retrieval import ranking
from src.retrieval.semantic import PlayerSemanticMatch, SemanticIndex
from src.schemas import METRICS, SearchPlan
from src.stats.queries import Eligibility, available_seasons, fetch_candidates, find_player

Method = Literal["stats", "semantic", "hybrid"]
PASSAGES_PER_PLAYER = 2


@dataclass
class RankingSettings:
    stat_weight: float = 0.6
    semantic_weight: float = 0.4
    missing_semantic_policy: Literal["zero", "neutral"] = "zero"
    eligibility: Eligibility = field(default_factory=Eligibility)
    limit: int = 10


@dataclass
class Passage:
    document_id: str
    text: str
    source_name: str
    source_url: str
    source_section: str | None
    published_at: str | None
    description_type: str
    license: str | None
    similarity: float


@dataclass
class PlayerResult:
    rank: int
    player_id: str
    player_name: str
    stats: dict
    breakdown: ranking.ScoreBreakdown
    passages: list[Passage]


@dataclass
class SearchResult:
    plan: SearchPlan
    method: Method
    results: list[PlayerResult]
    candidate_count: int
    messages: list[str]
    timings_ms: dict[str, float]
    reference_player_id: str | None = None


class ResearchEngine:
    def __init__(self, conn: sqlite3.Connection, index: SemanticIndex | None):
        self.conn = conn
        self.index = index

    def search(self, plan: SearchPlan, method: Method, settings: RankingSettings) -> SearchResult:
        timings: dict[str, float] = {}
        messages: list[str] = []

        def done(results: list[PlayerResult], candidate_count: int, reference_id: str | None = None) -> SearchResult:
            return SearchResult(plan, method, results, candidate_count, messages, timings, reference_id)

        seasons = available_seasons(self.conn)
        if plan.season not in seasons:
            messages.append(f"No statistics are loaded for {plan.season}. Available: {', '.join(seasons) or 'none'}.")
            return done([], 0)

        reference = self._resolve_reference_player(plan, messages)

        started = time.perf_counter()
        if method == "semantic":
            candidates = fetch_candidates(self.conn, plan.season)
        else:
            candidates = fetch_candidates(
                self.conn, plan.season, plan.positions, plan.hard_filters, settings.eligibility
            )
        if reference is not None:
            candidates = [c for c in candidates if c["player_id"] != reference["player_id"]]
        timings["sql"] = (time.perf_counter() - started) * 1000

        matches: dict[str, PlayerSemanticMatch] = {}
        if method in ("semantic", "hybrid"):
            started = time.perf_counter()
            matches = self._semantic_matches(plan, reference, messages)
            timings["vector_search"] = (time.perf_counter() - started) * 1000
            if method == "semantic":
                candidates = [c for c in candidates if c["player_id"] in matches]

        if not candidates:
            messages.append("No players satisfy the season, position and filter constraints.")
            return done([], 0, reference and reference["player_id"])

        started = time.perf_counter()
        stat_fits, goals = self._statistical_fit(plan, candidates, reference, method, messages)
        similarities = [matches[c["player_id"]].best_by_type if c["player_id"] in matches else None for c in candidates]

        if method == "stats" and all(s is None for s in stat_fits):
            messages.append("The question has no statistical goal or filter to rank on.")
            return done([], len(candidates), reference and reference["player_id"])
        if method == "hybrid" and all(s is None for s in stat_fits) and all(s is None for s in similarities):
            messages.append("The question has nothing to rank on: no statistical goals, filters or style terms.")
            return done([], len(candidates), reference and reference["player_id"])

        breakdowns = ranking.combine(
            [c["player_id"] for c in candidates],
            stat_fits,
            similarities,
            goals,
            settings.stat_weight,
            settings.semantic_weight,
            settings.missing_semantic_policy,
        )
        timings["ranking"] = (time.perf_counter() - started) * 1000

        if method == "hybrid" and matches:
            missing = sum(1 for s in similarities if s is None)
            if missing:
                messages.append(
                    f"{missing} of {len(candidates)} candidates have no style description in the corpus; "
                    f"their semantic score uses the '{settings.missing_semantic_policy}' policy."
                )

        by_id = {c["player_id"]: c for c in candidates}
        results = []
        for rank, breakdown in enumerate(breakdowns[: settings.limit], 1):
            row = by_id[breakdown.player_id]
            passages = self._passages(matches.get(breakdown.player_id)) if method != "stats" else []
            results.append(PlayerResult(rank, row["player_id"], row["player_name"], row, breakdown, passages))
        return done(results, len(candidates), reference and reference["player_id"])

    def _resolve_reference_player(self, plan: SearchPlan, messages: list[str]) -> dict | None:
        """Return the stats row of the player named in a "similar to X" query, if they are in the season."""
        if not plan.reference_player:
            return None
        found = find_player(self.conn, plan.reference_player, plan.season)
        if found is None:
            messages.append(f"Could not find {plan.reference_player!r} in the {plan.season} data.")
            return None
        return next(c for c in fetch_candidates(self.conn, plan.season) if c["player_id"] == found["player_id"])

    def _semantic_matches(self, plan: SearchPlan, reference: dict | None, messages: list[str]) -> dict[str, PlayerSemanticMatch]:
        if self.index is None or len(self.index) == 0:
            messages.append("The description index is not available, so no semantic evidence was used.")
            return {}
        if reference is not None:
            vector = self.index.player_vector(reference["player_id"])
            if vector is not None:
                matches = self.index.search_vector(vector)
                matches.pop(reference["player_id"], None)
                return matches
            messages.append(f"{reference['player_name']} has no description in the corpus; style similarity uses the question text.")
        if not plan.semantic_query.strip():
            messages.append("The question has no playing-style terms, so no semantic search was run.")
            return {}
        return self.index.search_text(plan.semantic_query)

    def _statistical_fit(
        self, plan: SearchPlan, candidates: list[dict], reference: dict | None, method: Method, messages: list[str]
    ) -> tuple[list[float | None], list[list[ranking.GoalBreakdown]] | None]:
        if method == "semantic":
            return [None] * len(candidates), None

        goals = ranking.goal_scores(candidates, plan.preferences) if plan.preferences else None
        fits = [ranking.statistical_fit(g) for g in goals] if goals else [None] * len(candidates)

        if reference is not None:
            profile = ranking.profile_similarity(candidates, reference)
            fits = [ranking.mean_or_none([f, p]) for f, p in zip(fits, profile)]
            label = f"statistical profile similarity to {reference['player_name']}"
            base = goals or [[] for _ in candidates]
            goals = [g + [ranking.GoalBreakdown(label, "high", p, [])] for g, p in zip(base, profile)]
        elif not plan.preferences and plan.hard_filters:
            # No stated preference: rank on the metrics the user filtered on.
            messages.append("No ranking preference was stated; players are ranked on the metrics in the filters.")
            directions = {f.metric: ("low" if f.op == "<=" else "high") for f in plan.hard_filters}
            goals = ranking.metric_goal_scores(candidates, directions)
            fits = [ranking.statistical_fit(g) for g in goals]
        return fits, goals

    def _passages(self, match: PlayerSemanticMatch | None) -> list[Passage]:
        if match is None:
            return []
        hits = match.hits[:PASSAGES_PER_PLAYER]
        placeholders = ", ".join("?" for _ in hits)
        rows = {
            r["document_id"]: r
            for r in self.conn.execute(f"SELECT * FROM player_documents WHERE document_id IN ({placeholders})", [h.document_id for h in hits])
        }
        return [
            Passage(
                h.document_id, rows[h.document_id]["text"], rows[h.document_id]["source_name"],
                rows[h.document_id]["source_url"], rows[h.document_id]["source_section"],
                rows[h.document_id]["published_at"], rows[h.document_id]["description_type"],
                rows[h.document_id]["license"], h.similarity,
            )
            for h in hits
            if h.document_id in rows
        ]


def format_stat(metric: str, value: float | None) -> str:
    """Single source of truth for how numbers are shown (UI, prompts and grounding checks)."""
    if value is None:
        return "not available"
    unit = METRICS[metric].unit
    if unit == "fraction":
        return f"{value * 100:.1f}%"
    if unit == "count":
        return f"{int(value)}"
    if unit == "feet":
        return f"{value:.1f} ft"
    return f"{value:.1f}"
