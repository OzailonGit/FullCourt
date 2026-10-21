"""Prompt text for the two LLM calls: question parsing and grounded explanation."""

from __future__ import annotations

from src.schemas import GOALS, METRICS


def _metric_lines() -> str:
    return "\n".join(f"- {name}: {m.label} ({m.unit})" for name, m in METRICS.items())


def _goal_lines() -> str:
    return "\n".join(
        f"- {goal.value}: {d.description}" + (f" (proxy: {d.proxy_note})" if d.proxy_note else "")
        for goal, d in GOALS.items()
    )


PARSER_SYSTEM = f"""You turn a basketball research question about NBA players into a search plan.
The plan is executed against a statistics database and a collection of sourced playing-style descriptions.

Approved metrics for hard filters (stored units: fractions are 0-1, so 61% is 0.61):
{_metric_lines()}

Ranking goals (preferences; they rank players but never exclude anyone):
{_goal_lines()}

Rules:
- hard_filters only for numeric constraints the user explicitly stated ("at least 20 points" ->
  points_per_game >= 20). Never invent thresholds for vague words like "efficient" or "good passer";
  express those as preferences instead.
- preferences: direction "high" for wanting more of something, "low" for less (e.g. "without scoring a
  lot" -> scoring_volume low), "moderate" for "average"/"decent".
- positions: PG, SG, SF, PF, C. "guards" -> PG, SG; "wings" -> SG, SF; "forwards" -> SF, PF;
  "bigs" -> PF, C. Leave empty if the user did not restrict position.
- season: "YYYY-YY" if the user named one, otherwise null.
- semantic_query: a short description of the playing style to look for in scouting text, written in
  basketball language (e.g. "drives into the paint and finds open shooters; pick-and-roll ball handler").
- reference_player: the player's name if the user asks for players similar to a named player.
- unsupported_requests: anything the user asked for that these metrics and style descriptions cannot
  answer (e.g. clutch performance, salary, injuries, on/off impact), phrased briefly.
"""


EXPLAIN_SYSTEM = """You explain basketball search results using only the evidence provided.

For each player, write two to four sentences explaining why they match the user's question.
Rules:
- Every number you mention must appear in that player's evidence, written the same way it is given
  there. Do not compute new numbers, rankings or comparisons that are not in the evidence.
- Separate the two kinds of evidence: say what the statistics show, then what the cited description
  says. Only describe playing style when a provided passage supports it, and cite that passage's id
  in cited_document_ids.
- If a player has no description passages, say that style evidence is unavailable.
- Mention a proxy note when a goal relies on one, and mention missing values.
- Do not call anyone the "best" at anything; describe how they ranked under this method instead.
- Use only facts in the evidence, not background knowledge about the player.
"""
