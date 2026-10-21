"""HTTP API for the React frontend.

Run with:  uvicorn src.api:app --reload

The API returns display-ready text (formatted stats, percentile wording) produced by the same
helpers the explanations and grounding check use, so the browser never formats numbers itself and
the UI can't drift from what the evidence says. When frontend/dist exists (after `npm run build`),
the built app is served from the same origin.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from src import pipeline
from src.generation.explanations import Explanation, goal_label, percentile_text
from src.retrieval.hybrid import Method, PlayerResult, ResearchEngine, format_stat
from src.schemas import METRICS
from src.stats.profiles import DESCRIPTION_TYPE as STAT_PROFILE_TYPE
from src.stats.queries import available_seasons

METHOD_LABELS: dict[str, str] = {
    "hybrid": "Hybrid (statistics + style profiles)",
    "stats": "Statistics only",
    "semantic": "Style profiles only",
}
EXAMPLES = [
    "Find efficient scoring guards who are good playmakers and can create their own shots.",
    "Rim-running bigs who catch lobs and protect the rim",
    "Three-and-D wings who space the floor and defend",
    "Find playmakers who create advantages without scoring a lot.",
    "Bigs who shoot at least 35% from three and grab 8+ rebounds",
    "Find players with a similar style to Jalen Brunson",
]
KEY_METRICS = ["points_per_game", "assists_per_game", "true_shooting_pct", "usage_rate", "assist_rate", "minutes_per_game"]
FRONTEND_DIST = config.ROOT / "frontend" / "dist"
STATS_TABLE_LABELS = {"per_game": "per game", "advanced": "advanced", "shooting": "shooting", "play-by-play": "play-by-play"}

app = FastAPI(title="FullCourt API")

# The engine shares one SQLite connection and one embedding model; searches are serialized
# because neither is safe to use from several request threads at once.
_search_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_engine() -> ResearchEngine:
    return pipeline.load_engine()


@lru_cache(maxsize=1)
def get_llm():
    return pipeline.load_llm()


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    method: Method = "hybrid"
    season: str | None = None
    use_eligibility: bool = True
    stat_weight: float = Field(default=config.STAT_WEIGHT, ge=0.0, le=1.0)
    missing_semantic_policy: Literal["zero", "neutral"] = config.MISSING_SEMANTIC_POLICY


@app.get("/api/meta")
def meta() -> dict:
    try:
        engine = get_engine()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    llm, llm_status = get_llm()
    return {
        "seasons": available_seasons(engine.conn),
        "default_season": config.DEFAULT_SEASON,
        "methods": [{"value": value, "label": label} for value, label in METHOD_LABELS.items()],
        "examples": EXAMPLES,
        "defaults": {
            "min_games": config.DEFAULT_MIN_GAMES,
            "min_minutes": config.DEFAULT_MIN_MINUTES,
            "stat_weight": config.STAT_WEIGHT,
            "missing_semantic_policy": config.MISSING_SEMANTIC_POLICY,
        },
        "llm": {"available": llm is not None, "status": llm_status},
        "index_size": len(engine.index) if engine.index else 0,
    }


@app.post("/api/search")
def search(request: SearchRequest) -> dict:
    try:
        engine = get_engine()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    llm, _ = get_llm()

    settings = pipeline.default_settings(request.use_eligibility)
    settings.stat_weight, settings.semantic_weight = request.stat_weight, 1.0 - request.stat_weight
    settings.missing_semantic_policy = request.missing_semantic_policy

    with _search_lock:
        try:
            output = pipeline.run(request.question, engine, request.method, settings, llm, season=request.season)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    result = output.result
    explanations = {e.player_id: e for e in output.explanations}
    return {
        "summary": {
            "method": result.method,
            "method_label": METHOD_LABELS[result.method],
            "season": result.plan.season,
            "season_was_defaulted": result.plan.season_was_defaulted,
            "parser": result.plan.parser,
            "candidate_count": result.candidate_count,
            "eligibility_applied": request.use_eligibility and result.method != "semantic",
        },
        "plan": result.plan.model_dump(mode="json"),
        "messages": result.messages,
        "warnings": output.warnings,
        "unsupported": result.plan.unsupported_requests,
        "timings_ms": {stage: round(ms, 1) for stage, ms in result.timings_ms.items()},
        "results": [
            _player_payload(player, explanations.get(player.player_id), settings.stat_weight)
            for player in result.results
        ],
    }


def _stats_tables(source_url: str) -> list[dict]:
    """Turn the stored source string into labelled links.

    scripts/import_stats.py stores every table URL in one column joined with " ; ". Splitting it here
    keeps that storage detail out of the browser.
    """
    tables = []
    for url in (part.strip() for part in source_url.split(";")):
        if not url:
            continue
        match = re.search(r"_([a-z_-]+)\.html$", url)
        kind = match.group(1) if match else ""
        tables.append({"label": STATS_TABLE_LABELS.get(kind, kind.replace("_", " ") or "source"), "url": url})
    return tables


def _score_contributions(player: PlayerResult, stat_weight: float) -> tuple[float, float]:
    """Split the final score into the parts that came from statistics and from style match.

    Mirrors ranking.combine: a component the query didn't use carries no weight, and the other
    then carries all of it.
    """
    b = player.breakdown
    used = set(b.components_used)
    if used == {"statistics"}:
        return b.final_score, 0.0
    if used == {"semantic"}:
        return 0.0, b.final_score
    return stat_weight * (b.stat_fit or 0.0), (1.0 - stat_weight) * (b.semantic_score or 0.0)


def _player_payload(player: PlayerResult, explanation: Explanation | None, stat_weight: float) -> dict:
    stats = player.stats
    b = player.breakdown
    stat_part, style_part = _score_contributions(player, stat_weight)
    return {
        "rank": player.rank,
        "player_id": player.player_id,
        "player_name": player.player_name,
        "position": stats.get("position"),
        "team": stats.get("team"),
        "teams": stats.get("teams"),
        "season": stats["season"],
        "games": format_stat("games_played", stats.get("games_played")),
        "key_stats": [
            {"metric": m, "label": METRICS[m].label, "value": format_stat(m, stats.get(METRICS[m].column))}
            for m in KEY_METRICS
        ],
        "score": {
            "final": b.final_score,
            "stat_fit": b.stat_fit,
            "style_match": b.semantic_score,
            "style_missing": b.semantic_missing,
            "components_used": b.components_used,
            "stat_contribution": stat_part,
            "style_contribution": style_part,
        },
        "goals": [
            {
                "label": goal_label(g.goal, g.direction),
                "score": g.score,
                "proxy_note": g.proxy_note,
                "metrics": [
                    {
                        "label": METRICS[m.metric].label,
                        "value": format_stat(m.metric, m.value),
                        "pool_rank": percentile_text(m.percentile, m.lower_is_better),
                    }
                    for m in g.metrics
                ],
            }
            for g in b.goals
        ],
        "explanation": None if explanation is None else {
            "text": explanation.text,
            "source": explanation.source,
            "grounding_issues": explanation.grounding_issues,
        },
        "passages": [
            {
                "document_id": p.document_id,
                "text": p.text,
                "source_name": p.source_name,
                "source_url": p.source_url,
                "source_section": p.source_section,
                "description_type": p.description_type,
                "is_stat_profile": p.description_type == STAT_PROFILE_TYPE,
                "license": p.license,
                "published_at": p.published_at,
            }
            for p in player.passages
        ],
        "stats_source": {"name": stats["source_name"], "tables": _stats_tables(stats["source_url"])},
    }


if Path(FRONTEND_DIST).exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
