"""Build per-player evidence packages and turn them into grounded explanations.

With an LLM configured, the model receives only the evidence packages and must return structured
explanations. Each explanation then goes through check_grounding(): every number it states must
appear in that player's evidence, and every cited document id must be one of that player's passages.
Explanations that fail are replaced by the deterministic template and the failure is reported.
Without an LLM, the template is used for every player.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from src.llm.client import LLMClient, LLMError
from src.llm.prompts import EXPLAIN_SYSTEM
from src.retrieval.hybrid import PlayerResult, SearchResult, format_stat
from src.schemas import GOALS, METRICS, Goal
from src.stats.profiles import DESCRIPTION_TYPE as STAT_PROFILE_TYPE

CORE_METRICS = ["games_played", "minutes_per_game", "points_per_game", "assists_per_game", "true_shooting_pct"]
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


class PlayerExplanation(BaseModel):
    player_id: str
    explanation: str
    cited_document_ids: list[str]


class ExplanationBatch(BaseModel):
    players: list[PlayerExplanation]


@dataclass
class Explanation:
    player_id: str
    text: str
    cited_document_ids: list[str]
    source: Literal["llm", "template"]
    grounding_issues: list[str] = field(default_factory=list)


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def percentile_text(p: float | None, lower_is_better: bool = False) -> str:
    if p is None:
        return "not available"
    text = f"{ordinal(round(p * 100))} percentile"
    return f"{text}, lower is better here" if lower_is_better else text


def goal_label(goal: str, direction: str) -> str:
    if goal.startswith("metric:"):
        name = METRICS[goal.removeprefix("metric:")].label
    elif goal in Goal.__members__:
        name = GOALS[Goal(goal)].description.lower()
    else:
        name = goal
    return name if direction == "high" else f"{name} ({direction})"


def evidence_package(result: SearchResult, player: PlayerResult) -> dict:
    """Everything the explanation may rely on, already formatted the way it should be quoted."""
    b = player.breakdown
    metrics_shown = list(CORE_METRICS)
    for g in b.goals:
        metrics_shown += [m.metric for m in g.metrics if m.metric not in metrics_shown]
    for f in result.plan.hard_filters:
        if f.metric not in metrics_shown:
            metrics_shown.append(f.metric)

    return {
        "player_id": player.player_id,
        "player_name": player.player_name,
        "rank": player.rank,
        "season": player.stats["season"],
        "team": player.stats.get("team"),
        "position": player.stats.get("position"),
        "statistics": {METRICS[m].label: format_stat(m, player.stats.get(METRICS[m].column)) for m in metrics_shown},
        "ranking": {
            "method": result.method,
            "candidates_considered": result.candidate_count,
            "components_used": b.components_used,
            "final_score": f"{b.final_score:.2f}",
            "statistical_fit": None if b.stat_fit is None else f"{b.stat_fit:.2f}",
            "semantic_relevance": None if b.semantic_score is None else f"{b.semantic_score:.2f}",
            "style_description_available": not b.semantic_missing,
        },
        "goals": [
            {
                "goal": goal_label(g.goal, g.direction),
                "score": None if g.score is None else f"{g.score:.2f}",
                "metrics": {
                    METRICS[m.metric].label: {"value": format_stat(m.metric, m.value), "pool_rank": percentile_text(m.percentile, m.lower_is_better)}
                    for m in g.metrics
                },
                **({"proxy_note": g.proxy_note} if g.proxy_note else {}),
            }
            for g in b.goals
        ],
        "passages": [
            {
                "document_id": p.document_id,
                "source": p.source_name,
                "section": p.source_section,
                "type": p.description_type,
                "similarity": f"{p.similarity:.2f}",
                "text": p.text,
            }
            for p in player.passages
        ],
    }


def _sentence_case(text: str) -> str:
    """Capitalize the first letter only (str.capitalize would lowercase player names)."""
    return text[:1].upper() + text[1:]


def template_explanation(result: SearchResult, player: PlayerResult) -> Explanation:
    b = player.breakdown
    parts = [f"Ranked #{player.rank} of {result.candidate_count} candidates by the {result.method} method."]
    for g in b.goals:
        if g.metrics:
            facts = ", ".join(
                f"{METRICS[m.metric].label} {format_stat(m.metric, m.value)} ({percentile_text(m.percentile, m.lower_is_better)})"
                for m in g.metrics
            )
            parts.append(f"{_sentence_case(goal_label(g.goal, g.direction))}: {facts}.")
        elif g.score is not None:
            parts.append(f"{_sentence_case(g.goal)}: {g.score:.2f}.")
        if g.proxy_note:
            parts.append(f"Note: {g.proxy_note}")
    cited = []
    if result.method == "stats":
        parts.append("Style descriptions are not used by the statistics-only method.")
    elif player.passages:
        top = player.passages[0]
        if top.description_type == STAT_PROFILE_TYPE:
            parts.append(
                f"Style evidence: the closest match is this player's stat-based profile (written from the "
                f"numbers, not scouting text), similarity {top.similarity:.2f}; read it below to judge the match."
            )
        else:
            parts.append(
                f"Style evidence: the closest passage is from {top.source_name} ({top.source_section}), "
                f"similarity {top.similarity:.2f}; read it below to judge the match."
            )
        cited.append(top.document_id)
    else:
        parts.append("Style evidence: no description of this player is available in the corpus.")
    return Explanation(player.player_id, " ".join(parts), cited, "template")


def allowed_numbers(package: dict, question: str) -> set[str]:
    text = json.dumps(package, ensure_ascii=False) + " " + question
    return {_canonical(n) for n in NUMBER.findall(text)}


def _canonical(number: str) -> str:
    return str(float(number))


def check_grounding(explanation: PlayerExplanation, package: dict, question: str) -> list[str]:
    issues = []
    allowed = allowed_numbers(package, question)
    unsupported = sorted({n for n in NUMBER.findall(explanation.explanation) if _canonical(n) not in allowed})
    if unsupported:
        issues.append(f"numbers not in the evidence: {', '.join(unsupported)}")
    passage_ids = {p["document_id"] for p in package["passages"]}
    bad_citations = [d for d in explanation.cited_document_ids if d not in passage_ids]
    if bad_citations:
        issues.append(f"cited passages that were not provided: {', '.join(bad_citations)}")
    if re.search(r"\bthe best\b|\bbest (?:player|scorer|passer|shooter|playmaker)\b", explanation.explanation, re.I):
        issues.append("uses 'best' wording that the ranking does not justify")
    return issues


def explain(result: SearchResult, question: str, llm: LLMClient | None) -> tuple[list[Explanation], list[str]]:
    """Return one explanation per result, plus warnings to show the user."""
    if not result.results:
        return [], []
    templates = {p.player_id: template_explanation(result, p) for p in result.results}
    if llm is None:
        return list(templates.values()), []

    packages = {p.player_id: evidence_package(result, p) for p in result.results}
    prompt = json.dumps(
        {
            "question": question,
            "search_plan": result.plan.model_dump(mode="json", exclude={"parser"}),
            "players": list(packages.values()),
        },
        ensure_ascii=False,
        indent=1,
    )
    try:
        batch = llm.parse(EXPLAIN_SYSTEM, prompt, ExplanationBatch, effort="medium")
    except LLMError as e:
        return list(templates.values()), [f"LLM explanation failed ({e}); showing template explanations."]

    generated = {e.player_id: e for e in batch.players}
    explanations, warnings = [], []
    for player in result.results:
        candidate = generated.get(player.player_id)
        if candidate is None:
            explanations.append(templates[player.player_id])
            continue
        issues = check_grounding(candidate, packages[player.player_id], question)
        if issues:
            fallback = templates[player.player_id]
            fallback.grounding_issues = issues
            explanations.append(fallback)
            warnings.append(f"{player.player_name}: generated explanation failed grounding ({'; '.join(issues)}); template shown.")
        else:
            explanations.append(Explanation(player.player_id, candidate.explanation, candidate.cited_document_ids, "llm"))
    return explanations, warnings
