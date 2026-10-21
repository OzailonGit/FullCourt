import pytest
from pydantic import ValidationError

from src.llm.client import LLMError
from src.query_parser import parse_question, parse_rules
from src.schemas import Goal, HardFilter, LLMSearchPlan, Position, Preference, SearchPlan

SEASON = "2025-26"


def goals(plan):
    return {p.goal: p.direction for p in plan.preferences}


def test_unsupported_metric_is_rejected():
    with pytest.raises(ValidationError):
        HardFilter(metric="clutch_points; DROP TABLE players", op=">=", value=1)


def test_percentage_filter_converted_to_fraction():
    assert HardFilter(metric="three_point_pct", op=">=", value=40).value == pytest.approx(0.40)
    assert HardFilter(metric="three_point_pct", op=">=", value=0.4).value == pytest.approx(0.40)


def test_season_format_validated():
    with pytest.raises(ValidationError):
        SearchPlan(season="last year", semantic_query="x")


def test_vague_terms_become_preferences_not_thresholds():
    plan = parse_rules("Find efficient scoring guards who are good playmakers and can create their own shots.", SEASON).plan
    assert plan.hard_filters == []
    assert set(plan.positions) == {Position.PG, Position.SG}
    assert goals(plan) == {
        Goal.scoring_efficiency: "high",
        Goal.scoring_volume: "high",
        Goal.playmaking: "high",
        Goal.shot_creation: "high",
    }


def test_explicit_numbers_become_hard_filters_with_carried_operator():
    plan = parse_rules("Guards with at least 20 points and 5 assists per game", SEASON).plan
    assert [(f.metric, f.op, f.value) for f in plan.hard_filters] == [
        ("points_per_game", ">=", 20.0),
        ("assists_per_game", ">=", 5.0),
    ]
    # filter terms are not double counted as ranking preferences, and nothing stylistic is left
    assert plan.preferences == []
    assert plan.semantic_query == ""


def test_upper_bounds_and_plus_syntax():
    plan = parse_rules("Bigs with 8+ rebounds and under 2 turnovers", SEASON).plan
    assert [(f.metric, f.op, f.value) for f in plan.hard_filters] == [
        ("rebounds_per_game", ">=", 8.0),
        ("turnovers_per_game", "<=", 2.0),
    ]


def test_bare_number_without_comparison_is_not_enforced():
    plan = parse_rules("guards who average 25 points", SEASON).plan
    assert plan.hard_filters == []


def test_low_and_moderate_directions():
    assert goals(parse_rules("Find playmakers who create advantages without scoring a lot.", SEASON).plan)[Goal.scoring_volume] == "low"
    assert goals(parse_rules("Find high scorers who are average playmakers.", SEASON).plan)[Goal.playmaking] == "moderate"


def test_season_defaulted_and_explicit():
    assert parse_rules("efficient scorers", SEASON).plan.season_was_defaulted
    plan = parse_rules("efficient scorers in 2024-25", SEASON).plan
    assert plan.season == "2024-25" and not plan.season_was_defaulted


def test_reference_player_and_unsupported_requests():
    plan = parse_rules("Find players with a similar style to Jalen Brunson who are clutch", SEASON).plan
    assert plan.reference_player == "Jalen Brunson"
    assert any("clutch" in u for u in plan.unsupported_requests)


def test_similar_without_name_warns():
    result = parse_rules("Find players with a similar style to this guard", SEASON)
    assert result.plan.reference_player is None
    assert result.warnings


def test_empty_question_rejected():
    with pytest.raises(ValueError):
        parse_question("   ", SEASON)


class FailingLLM:
    name = "failing"

    def parse(self, system, user, schema, effort="low"):
        raise LLMError("boom")


class FixedLLM:
    name = "fixed"

    def parse(self, system, user, schema, effort="low"):
        return LLMSearchPlan(
            positions=[Position.PG],
            preferences=[Preference(goal=Goal.playmaking), Preference(goal=Goal.playmaking, direction="low")],
            semantic_query="pass-first point guard",
        )


def test_llm_failure_falls_back_to_rules():
    result = parse_question("efficient scorers", SEASON, FailingLLM())
    assert result.plan.parser == "rules"
    assert "LLM parsing failed" in result.warnings[0]


def test_llm_plan_is_validated_defaulted_and_deduplicated():
    plan = parse_question("pass-first point guards", SEASON, FixedLLM()).plan
    assert plan.parser == "llm"
    assert plan.season == SEASON and plan.season_was_defaulted
    assert [(p.goal, p.direction) for p in plan.preferences] == [(Goal.playmaking, "high")]


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Three-and-D wings who space the floor and defend", {Goal.spot_up_shooting, Goal.defense_activity, Goal.three_point_shooting}),
        ("Rim-running bigs who catch lobs", {Goal.rim_finishing}),
        ("Guards who attack the rim and get to the line", {Goal.rim_pressure}),
        ("Post scorers who hit mid-range jumpers", {Goal.scoring_volume, Goal.post_scoring, Goal.midrange_scoring}),
    ],
)
def test_style_goals_from_play_type_language(question, expected):
    assert set(goals(parse_rules(question, SEASON).plan)) == expected
