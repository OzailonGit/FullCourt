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
    unit: Literal["per_game", "fraction", "count", "points_per_100", "feet"]
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
    "average_shot_distance": Metric("average_shot_distance", "Average shot distance", "feet"),
    "shot_share_0_3ft": Metric("shot_share_0_3ft", "Share of shots at 0-3 ft", "fraction"),
    "shot_share_3_10ft": Metric("shot_share_3_10ft", "Share of shots at 3-10 ft", "fraction"),
    "shot_share_10_16ft": Metric("shot_share_10_16ft", "Share of shots at 10-16 ft", "fraction"),
    "shot_share_16ft_to_3pt": Metric("shot_share_16ft_to_3pt", "Share of shots from 16 ft to the arc", "fraction"),
    "three_point_attempt_rate": Metric("three_point_attempt_rate", "Share of shots that are threes", "fraction"),
    "fg_pct_0_3ft": Metric("fg_pct_0_3ft", "FG% at 0-3 ft", "fraction"),
    "fg_pct_3_10ft": Metric("fg_pct_3_10ft", "FG% at 3-10 ft", "fraction"),
    "fg_pct_10_16ft": Metric("fg_pct_10_16ft", "FG% at 10-16 ft", "fraction"),
    "fg_pct_16ft_to_3pt": Metric("fg_pct_16ft_to_3pt", "FG% from 16 ft to the arc", "fraction"),
    "assisted_share_2pt": Metric("assisted_share_2pt", "Share of made 2s assisted", "fraction"),
    "assisted_share_3pt": Metric("assisted_share_3pt", "Share of made 3s assisted", "fraction"),
    "dunk_share": Metric("dunk_share", "Share of shots that are dunks", "fraction"),
    "corner_three_share": Metric("corner_three_share", "Share of 3PA from the corners", "fraction"),
    "corner_three_pct": Metric("corner_three_pct", "Corner 3P%", "fraction"),
    "minutes_share_pg": Metric("minutes_share_pg", "Minutes at PG (estimated)", "fraction"),
    "minutes_share_sg": Metric("minutes_share_sg", "Minutes at SG (estimated)", "fraction"),
    "minutes_share_sf": Metric("minutes_share_sf", "Minutes at SF (estimated)", "fraction"),
    "minutes_share_pf": Metric("minutes_share_pf", "Minutes at PF (estimated)", "fraction"),
    "minutes_share_c": Metric("minutes_share_c", "Minutes at C (estimated)", "fraction"),
    "shooting_fouls_drawn_per_game": Metric("shooting_fouls_drawn_per_game", "Shooting fouls drawn per game", "per_game"),
    "and_ones_per_game": Metric("and_ones_per_game", "And-ones per game", "per_game"),
    "charges_drawn_per_game": Metric("charges_drawn_per_game", "Charges drawn per game", "per_game"),
    "bad_pass_turnovers_per_game": Metric(
        "bad_pass_turnovers_per_game", "Bad-pass turnovers per game", "per_game", higher_is_better=False
    ),
    "lost_ball_turnovers_per_game": Metric(
        "lost_ball_turnovers_per_game", "Lost-ball turnovers per game", "per_game", higher_is_better=False
    ),
    "points_created_by_assists_per_game": Metric(
        "points_created_by_assists_per_game", "Points created by assists per game", "per_game"
    ),
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
    rim_pressure = "rim_pressure"
    rim_finishing = "rim_finishing"
    spot_up_shooting = "spot_up_shooting"
    post_scoring = "post_scoring"
    midrange_scoring = "midrange_scoring"


@dataclass(frozen=True)
class GoalDefinition:
    description: str
    # Maps each metric name to its weight within the goal. Weights within a goal sum to 1, so correlated metrics
    # that measure the same construct (e.g. AST and AST%) are averaged, not double-counted.
    metrics: dict[str, float]
    proxy_note: str | None = None
    # Metrics where a lower value fits this goal better. Direction depends on the goal, not the metric:
    # a low assisted share means self-creation, while a high one means a catch-and-shoot role.
    lower_is_better: frozenset[str] = frozenset()


GOALS: dict[Goal, GoalDefinition] = {
    Goal.scoring_volume: GoalDefinition("Scores a lot of points", {"points_per_game": 1.0}),
    Goal.scoring_efficiency: GoalDefinition("Scores efficiently", {"true_shooting_pct": 1.0}),
    Goal.playmaking: GoalDefinition(
        "Creates baskets for teammates", {"assist_rate": 0.5, "assists_per_game": 0.5}
    ),
    Goal.ball_security: GoalDefinition("Avoids turnovers", {"turnover_rate": 1.0}),
    Goal.shot_creation: GoalDefinition(
        "Creates their own shot",
        {"assisted_share_2pt": 0.35, "assisted_share_3pt": 0.25, "usage_rate": 0.4},
        proxy_note="Unassisted makes also include putbacks and transition finishes, so usage rate is blended in "
        "to favor players who create on the ball.",
        lower_is_better=frozenset({"assisted_share_2pt", "assisted_share_3pt"}),
    ),
    Goal.three_point_shooting: GoalDefinition(
        "Shoots threes accurately and often", {"three_point_pct": 0.5, "three_point_attempts_per_game": 0.5}
    ),
    Goal.rebounding: GoalDefinition("Rebounds", {"rebound_rate": 0.5, "rebounds_per_game": 0.5}),
    Goal.defense_activity: GoalDefinition(
        "Generates steals, blocks and charges",
        {"steal_rate": 0.4, "block_rate": 0.4, "charges_drawn_per_game": 0.2},
        proxy_note="Steals, blocks and charges capture defensive activity, not overall defensive impact.",
    ),
    Goal.rim_pressure: GoalDefinition(
        "Attacks the rim and draws fouls",
        {"shot_share_0_3ft": 0.4, "shooting_fouls_drawn_per_game": 0.4, "and_ones_per_game": 0.2},
    ),
    Goal.rim_finishing: GoalDefinition(
        "Finishes at the rim off teammates' passes (rim runner, lob threat)",
        {"dunk_share": 0.4, "fg_pct_0_3ft": 0.3, "assisted_share_2pt": 0.3},
        proxy_note="There is no play-type data; dunks and assisted finishes near the rim stand in for "
        "rolls and lobs.",
    ),
    Goal.spot_up_shooting: GoalDefinition(
        "Spot-up and catch-and-shoot three-point shooting",
        {"assisted_share_3pt": 0.3, "three_point_pct": 0.3, "corner_three_share": 0.2, "three_point_attempt_rate": 0.2},
    ),
    Goal.post_scoring: GoalDefinition(
        "Scores in the post and close to the basket",
        {"shot_share_3_10ft": 0.6, "fg_pct_3_10ft": 0.4},
        proxy_note="There is no post-up data; shots from 3-10 ft also include floaters and short hooks.",
    ),
    Goal.midrange_scoring: GoalDefinition(
        "Scores from mid-range",
        {"shot_share_10_16ft": 0.3, "shot_share_16ft_to_3pt": 0.3, "fg_pct_10_16ft": 0.2, "fg_pct_16ft_to_3pt": 0.2},
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
