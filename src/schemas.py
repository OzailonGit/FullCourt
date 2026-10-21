"""Validated search-plan models and the fixed registry of approved metrics and ranking goals.

The LLM (or the rule-based fallback parser) produces a SearchPlan. Nothing in a SearchPlan is ever
interpolated into SQL directly: metrics are looked up in METRICS, which maps approved names to
column names defined in src/stats/database.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Position(str, Enum):
    PG = "PG"
    SG = "SG"
    SF = "SF"
    PF = "PF"
    C = "C"


@dataclass(frozen=True)
class Metric:
    column: str
    label: str
    unit: Literal["per_game", "fraction", "count", "points_per_100"]
    higher_is_better: bool = True


# Maps approved metric names to SQL columns. The only path from a search plan to SQL.
METRICS: dict[str, Metric] = {
    "games_played": Metric("games_played", "Games played", "count"),
    "minutes_per_game": Metric("minutes_per_game", "Minutes per game", "per_game"),
    "points_per_game": Metric("points_per_game", "Points per game", "per_game"),
    "assists_per_game": Metric("assists_per_game", "Assists per game", "per_game"),
    "rebounds_per_game": Metric("rebounds_per_game", "Rebounds per game", "per_game"),
    "steals_per_game": Metric("steals_per_game", "Steals per game", "per_game"),
    "blocks_per_game": Metric("blocks_per_game", "Blocks per game", "per_game"),
    "turnovers_per_game": Metric("turnovers_per_game", "Turnovers per game", "per_game", higher_is_better=False),
    "three_point_attempts_per_game": Metric("three_point_attempts_per_game", "3PA per game", "per_game"),
    "field_goal_pct": Metric("field_goal_pct", "FG%", "fraction"),
    "three_point_pct": Metric("three_point_pct", "3P%", "fraction"),
    "free_throw_pct": Metric("free_throw_pct", "FT%", "fraction"),
    "effective_field_goal_pct": Metric("effective_field_goal_pct", "eFG%", "fraction"),
    "true_shooting_pct": Metric("true_shooting_pct", "TS%", "fraction"),
    "usage_rate": Metric("usage_rate", "Usage rate", "fraction"),
    "assist_rate": Metric("assist_rate", "Assist rate (AST%)", "fraction"),
    "turnover_rate": Metric("turnover_rate", "Turnover rate (TOV%)", "fraction", higher_is_better=False),
    "rebound_rate": Metric("rebound_rate", "Rebound rate (TRB%)", "fraction"),
    "steal_rate": Metric("steal_rate", "Steal rate (STL%)", "fraction"),
    "block_rate": Metric("block_rate", "Block rate (BLK%)", "fraction"),
    "box_plus_minus": Metric("box_plus_minus", "Box Plus/Minus", "points_per_100"),
}


class Goal(str, Enum):
    scoring_volume = "scoring_volume"
    scoring_efficiency = "scoring_efficiency"
    playmaking = "playmaking"
    ball_security = "ball_security"
    shot_creation = "shot_creation"
    three_point_shooting = "three_point_shooting"
    rebounding = "rebounding"
    defense_activity = "defense_activity"


@dataclass(frozen=True)
class GoalDefinition:
    description: str
    # Maps each metric name to its weight within the goal. Weights within a goal sum to 1, so correlated metrics
    # that measure the same construct (e.g. AST and AST%) are averaged, not double-counted.
    metrics: dict[str, float]
    proxy_note: str | None = None


GOALS: dict[Goal, GoalDefinition] = {
    Goal.scoring_volume: GoalDefinition("Scores a lot of points", {"points_per_game": 1.0}),
    Goal.scoring_efficiency: GoalDefinition("Scores efficiently", {"true_shooting_pct": 1.0}),
    Goal.playmaking: GoalDefinition(
        "Creates baskets for teammates", {"assist_rate": 0.5, "assists_per_game": 0.5}
    ),
    Goal.ball_security: GoalDefinition("Avoids turnovers", {"turnover_rate": 1.0}),
    Goal.shot_creation: GoalDefinition(
        "Creates their own shot",
        {"usage_rate": 1.0},
        proxy_note="The dataset has no unassisted-shot data; usage rate is used as a proxy for on-ball creation load.",
    ),
    Goal.three_point_shooting: GoalDefinition(
        "Shoots threes accurately and often", {"three_point_pct": 0.5, "three_point_attempts_per_game": 0.5}
    ),
    Goal.rebounding: GoalDefinition("Rebounds", {"rebound_rate": 0.5, "rebounds_per_game": 0.5}),
    Goal.defense_activity: GoalDefinition(
        "Generates steals and blocks",
        {"steal_rate": 0.5, "block_rate": 0.5},
        proxy_note="Box-score defense (steals, blocks) captures activity, not overall defensive impact.",
    ),
}


MetricName = Literal[tuple(METRICS)]  # type: ignore[valid-type]


class HardFilter(BaseModel):
    """An explicit numeric constraint the user stated. Enforced in SQL before ranking."""

    metric: MetricName = Field(description="One of the approved metric names.")
    op: Literal[">=", "<="]
    value: float = Field(description="Value in the metric's stored unit. Percentages are fractions (61% -> 0.61).")

    @model_validator(mode="after")
    def fraction_units(self) -> HardFilter:
        if METRICS[self.metric].unit == "fraction" and self.value > 1:
            # "at least 40% from three" arrives as 40; stored values are fractions.
            self.value = self.value / 100.0
        return self


class Preference(BaseModel):
    """A ranking preference. Never filters anyone out."""

    goal: Goal
    direction: Literal["high", "low", "moderate"] = "high"


class SearchPlan(BaseModel):
    season: str
    season_was_defaulted: bool = False
    positions: list[Position] = Field(default_factory=list, description="Empty means any position.")
    hard_filters: list[HardFilter] = Field(default_factory=list)
    preferences: list[Preference] = Field(default_factory=list)
    semantic_query: str = Field(description="Playing-style description to search for in scouting text.")
    reference_player: str | None = Field(
        default=None, description="Player name when the user asks for players similar to someone."
    )
    unsupported_requests: list[str] = Field(
        default_factory=list, description="Parts of the question the dataset cannot answer, in plain words."
    )
    parser: Literal["llm", "rules"] = "rules"

    @field_validator("season")
    @classmethod
    def season_format(cls, season: str) -> str:
        import re

        if not re.fullmatch(r"\d{4}-\d{2}", season):
            raise ValueError("season must look like '2025-26'")
        return season


class LLMSearchPlan(BaseModel):
    """The subset of SearchPlan the LLM fills in. Kept flat and enum-heavy so outputs validate."""

    season: str | None = Field(default=None, description="Season like '2025-26', or null if the user did not say.")
    positions: list[Position] = Field(default_factory=list)
    hard_filters: list[HardFilter] = Field(default_factory=list)
    preferences: list[Preference] = Field(default_factory=list)
    semantic_query: str
    reference_player: str | None = None
    unsupported_requests: list[str] = Field(default_factory=list)
