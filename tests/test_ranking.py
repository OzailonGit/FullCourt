import pytest

from src.retrieval import ranking
from src.schemas import Goal, Preference


def test_percentile_ranks_handle_ties_nulls_and_single_values():
    assert ranking.percentile_ranks([10, 20, 20, 30]) == [0.0, 0.5, 0.5, 1.0]
    assert ranking.percentile_ranks([None, 5, 1]) == [None, 1.0, 0.0]
    assert ranking.percentile_ranks([7, None]) == [0.5, None]


def test_directions():
    assert ranking.apply_direction(0.9, "high") == pytest.approx(0.9)
    assert ranking.apply_direction(0.9, "low") == pytest.approx(0.1)
    assert ranking.apply_direction(0.5, "moderate") == pytest.approx(1.0)
    assert ranking.apply_direction(1.0, "moderate") == pytest.approx(0.0)


def test_lower_is_better_metrics_are_inverted():
    candidates = [{"turnover_rate": 0.08}, {"turnover_rate": 0.20}]
    scores = ranking.goal_scores(candidates, [Preference(goal=Goal.ball_security)])
    assert scores[0][0].score == pytest.approx(1.0)
    assert scores[1][0].score == pytest.approx(0.0)


def test_missing_metric_renormalizes_within_goal():
    candidates = [
        {"three_point_pct": None, "three_point_attempts_per_game": 8.0},
        {"three_point_pct": 0.40, "three_point_attempts_per_game": 2.0},
    ]
    scores = ranking.goal_scores(candidates, [Preference(goal=Goal.three_point_shooting)])
    # first player: only 3PA is known (top of pool), so the score is 1.0 rather than being dragged down by the NULL
    assert scores[0][0].score == pytest.approx(1.0)
    assert scores[0][0].metrics[0].value is None


def test_all_metrics_missing_gives_missing_goal_score():
    scores = ranking.goal_scores([{"true_shooting_pct": None}, {"true_shooting_pct": 0.6}], [Preference(goal=Goal.scoring_efficiency)])
    assert scores[0][0].score is None
    assert ranking.statistical_fit(scores[0]) is None


def test_combine_uses_weights_and_missing_semantic_policies():
    ids = ["a", "b", "c"]
    stat = [1.0, 0.5, 0.0]
    sims = [0.30, None, 0.70]  # b has no description

    zero = {r.player_id: r for r in ranking.combine(ids, stat, sims, None, 0.6, 0.4, "zero")}
    assert zero["a"].final_score == pytest.approx(0.6 * 1.0 + 0.4 * 0.0)
    assert zero["b"].semantic_missing and zero["b"].semantic_score == 0.0
    assert zero["c"].final_score == pytest.approx(0.4 * 1.0)

    neutral = {r.player_id: r for r in ranking.combine(ids, stat, sims, None, 0.6, 0.4, "neutral")}
    assert neutral["b"].semantic_score == pytest.approx(0.5)  # median of scaled [0.0, 1.0]


def test_combine_reweights_when_a_component_is_unavailable_for_the_query():
    results = ranking.combine(["a", "b"], [None, None], [0.2, 0.9], None, 0.6, 0.4)
    assert [r.player_id for r in results] == ["b", "a"]
    assert results[0].final_score == pytest.approx(1.0)
    assert results[0].components_used == ["semantic"]


def test_combine_is_sorted_and_deterministic_on_ties():
    results = ranking.combine(["b", "a"], [0.5, 0.5], [None, None], None, 0.6, 0.4)
    assert [r.player_id for r in results] == ["a", "b"]


def test_profile_similarity_prefers_closest_profile():
    reference = {"points_per_game": 30, "usage_rate": 0.33, "assist_rate": 0.40}
    near = {"points_per_game": 28, "usage_rate": 0.31, "assist_rate": 0.38}
    far = {"points_per_game": 6, "usage_rate": 0.12, "assist_rate": 0.05}
    sims = ranking.profile_similarity([near, far], reference)
    assert sims[0] > sims[1]


def test_shot_creation_favors_unassisted_scorers():
    creator = {"assisted_share_2pt": 0.10, "assisted_share_3pt": 0.30, "usage_rate": 0.30}
    finisher = {"assisted_share_2pt": 0.85, "assisted_share_3pt": 0.95, "usage_rate": 0.30}
    scores = ranking.goal_scores([creator, finisher], [Preference(goal=Goal.shot_creation)])
    assert scores[0][0].score > scores[1][0].score
    evidence = {m.metric: m for m in scores[0][0].metrics}
    assert evidence["assisted_share_2pt"].lower_is_better
    assert evidence["assisted_share_2pt"].percentile == pytest.approx(0.0)  # raw: lowest value in the pool


def test_spot_up_shooting_favors_assisted_threes():
    spot_up = {"assisted_share_3pt": 0.95, "three_point_pct": 0.38, "corner_three_share": 0.4, "three_point_attempt_rate": 0.7}
    pull_up = {"assisted_share_3pt": 0.30, "three_point_pct": 0.38, "corner_three_share": 0.05, "three_point_attempt_rate": 0.7}
    scores = ranking.goal_scores([spot_up, pull_up], [Preference(goal=Goal.spot_up_shooting)])
    assert scores[0][0].score > scores[1][0].score


def test_upper_bound_filter_ranks_fewer_turnovers_first():
    candidates = [{"turnovers_per_game": 1.0}, {"turnovers_per_game": 2.5}]
    scores = ranking.metric_goal_scores(candidates, {"turnovers_per_game": "low"})
    assert scores[0][0].score > scores[1][0].score
