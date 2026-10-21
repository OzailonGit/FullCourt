"""End-to-end pipeline: parse, then retrieve and rank, then explain. Shared by the UI and the evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import config
from src.generation.explanations import Explanation, explain
from src.llm.client import LLMClient, get_client
from src.query_parser import parse_question
from src.retrieval.hybrid import Method, RankingSettings, ResearchEngine, SearchResult
from src.retrieval.semantic import SemanticIndex, SentenceTransformerEmbedder
from src.stats.database import connect
from src.stats.queries import Eligibility


@dataclass
class PipelineOutput:
    result: SearchResult
    explanations: list[Explanation]
    warnings: list[str] = field(default_factory=list)


def default_settings(use_eligibility: bool = True) -> RankingSettings:
    return RankingSettings(
        stat_weight=config.STAT_WEIGHT,
        semantic_weight=config.SEMANTIC_WEIGHT,
        missing_semantic_policy=config.MISSING_SEMANTIC_POLICY,
        eligibility=Eligibility(config.DEFAULT_MIN_GAMES, config.DEFAULT_MIN_MINUTES) if use_eligibility else Eligibility(),
        limit=config.RESULT_LIMIT,
    )


@lru_cache(maxsize=1)
def load_index() -> SemanticIndex | None:
    if not config.INDEX_PATH.exists() or not config.INDEX_META_PATH.exists():
        return None
    embedder = SentenceTransformerEmbedder(config.EMBEDDING_MODEL, config.EMBEDDING_QUERY_PREFIX)
    return SemanticIndex.load(config.INDEX_PATH, config.INDEX_META_PATH, embedder)


def load_engine() -> ResearchEngine:
    if not config.DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {config.DB_PATH}. Run scripts/import_stats.py first.")
    return ResearchEngine(connect(config.DB_PATH), load_index())


def load_llm() -> tuple[LLMClient | None, str]:
    """Return (client or None, human-readable status for the UI)."""
    return get_client(config.LLM_PROVIDER, config.LLM_MODEL, config.OLLAMA_HOST, config.OLLAMA_NUM_CTX)


def run(
    question: str,
    engine: ResearchEngine,
    method: Method = "hybrid",
    settings: RankingSettings | None = None,
    llm: LLMClient | None = None,
    season: str | None = None,
    generate: bool = True,
) -> PipelineOutput:
    settings = settings or default_settings()

    started = time.perf_counter()
    parsed = parse_question(question, season or config.DEFAULT_SEASON, llm)
    plan = parsed.plan
    if season and plan.season_was_defaulted:
        plan.season = season
    parse_ms = (time.perf_counter() - started) * 1000

    result = engine.search(plan, method, settings)
    result.timings_ms = {"parse": parse_ms, **result.timings_ms}

    explanations, explain_warnings = [], []
    if generate:
        started = time.perf_counter()
        explanations, explain_warnings = explain(result, question, llm)
        result.timings_ms["generation"] = (time.perf_counter() - started) * 1000

    return PipelineOutput(result, explanations, parsed.warnings + explain_warnings)
