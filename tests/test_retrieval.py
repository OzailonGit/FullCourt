from src.retrieval.hybrid import RankingSettings
from src.schemas import Goal, HardFilter, Position, Preference, SearchPlan
from src.stats.queries import Eligibility, fetch_candidates

SEASON = "2025-26"


def plan(**kwargs):
    return SearchPlan(season=SEASON, semantic_query=kwargs.pop("semantic_query", ""), **kwargs)


def ids(result):
    return [r.player_id for r in result.results]


def test_hard_filters_and_positions_enforced_in_sql(conn):
    rows = fetch_candidates(conn, SEASON, [Position.PG], [HardFilter(metric="assists_per_game", op=">=", value=8.5)])
    assert [r["player_id"] for r in rows] == ["guard3"]


def test_null_never_satisfies_a_filter(conn):
    rows = fetch_candidates(conn, SEASON, hard_filters=[HardFilter(metric="three_point_pct", op="<=", value=0.99)])
    assert "big1" not in {r["player_id"] for r in rows}


def test_eligibility_floor(conn):
    rows = fetch_candidates(conn, SEASON, eligibility=Eligibility(min_games=20, min_minutes=10))
    assert "bench1" not in {r["player_id"] for r in rows}


def test_filter_values_are_bound_not_interpolated(conn):
    # A hostile-looking value can only ever be compared as a number/string, never executed.
    rows = fetch_candidates(conn, "2025-26' OR '1'='1")
    assert rows == []


def test_semantic_relevance_cannot_override_hard_constraint(engine):
    # wing1 is the best semantic match for isolation scoring but is filtered out by position.
    p = plan(
        positions=[Position.PG, Position.SG],
        preferences=[Preference(goal=Goal.shot_creation)],
        semantic_query="isolation scorer pull up jumpers off the dribble",
    )
    result = engine.search(p, "hybrid", RankingSettings())
    assert "wing1" not in ids(result)
    # ...while the semantic-only baseline, which ignores SQL filters, does return it first.
    assert ids(engine.search(p, "semantic", RankingSettings()))[0] == "wing1"


def test_stats_only_ranking_and_no_passages(engine):
    p = plan(preferences=[Preference(goal=Goal.playmaking)])
    result = engine.search(p, "stats", RankingSettings(eligibility=Eligibility(20, 10)))
    assert ids(result)[:2] == ["guard3", "guard1"]
    assert all(r.passages == [] for r in result.results)
    assert all(r.breakdown.components_used == ["statistics"] for r in result.results)


def test_hybrid_attaches_source_passages_and_flags_missing_descriptions(engine):
    p = plan(preferences=[Preference(goal=Goal.playmaking)], semantic_query="pass first point guard drives and kicks")
    result = engine.search(p, "hybrid", RankingSettings(eligibility=Eligibility(20, 10)))
    top = result.results[0]
    assert top.player_id == "guard3"
    assert top.passages and top.passages[0].document_id == "guard3-doc-0"
    assert top.passages[0].source_url == "fixture://doc"
    guard2 = next(r for r in result.results if r.player_id == "guard2")
    assert guard2.breakdown.semantic_missing and guard2.passages == []
    assert any("no style description" in m for m in result.messages)


def test_filters_only_question_ranks_on_filtered_metrics(engine):
    p = plan(hard_filters=[HardFilter(metric="points_per_game", op=">=", value=15)])
    result = engine.search(p, "hybrid", RankingSettings())
    assert ids(result) == ["guard1", "wing1", "guard2", "big1"]
    assert any("ranked on the metrics in the filters" in m for m in result.messages)


def test_empty_retrieval_reports_reason(engine):
    p = plan(hard_filters=[HardFilter(metric="points_per_game", op=">=", value=60)], preferences=[Preference(goal=Goal.playmaking)])
    result = engine.search(p, "hybrid", RankingSettings())
    assert result.results == []
    assert any("No players satisfy" in m for m in result.messages)


def test_unknown_season_reports_available_seasons(engine):
    result = engine.search(SearchPlan(season="1999-00", semantic_query="x"), "hybrid", RankingSettings())
    assert result.results == []
    assert "2025-26" in result.messages[0]


def test_nothing_to_rank_on(engine):
    result = engine.search(plan(), "stats", RankingSettings())
    assert result.results == [] and result.messages


def test_reference_player_is_excluded_and_drives_similarity(engine):
    p = plan(reference_player="Fixture Guard One")
    result = engine.search(p, "hybrid", RankingSettings(eligibility=Eligibility(20, 10)))
    assert result.reference_player_id == "guard1"
    assert "guard1" not in ids(result)
    # Style similarity comes from guard1's own description: guard3 shares 'drives' and 'open shooters'.
    best_style = max(result.results, key=lambda r: r.breakdown.semantic_similarity or -1)
    assert best_style.player_id == "guard3"
    assert all(r.breakdown.goals[-1].goal.startswith("statistical profile similarity") for r in result.results)
