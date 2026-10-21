import pytest
from fastapi.testclient import TestClient

from src import api


@pytest.fixture
def client(engine, monkeypatch):
    monkeypatch.setattr(api, "get_engine", lambda: engine)
    monkeypatch.setattr(api, "get_llm", lambda: (None, "LLM disabled (FULLCOURT_LLM_PROVIDER=none)"))
    return TestClient(api.app)


def test_meta_lists_seasons_methods_and_defaults(client):
    meta = client.get("/api/meta").json()
    assert meta["seasons"] == ["2025-26"]
    assert [m["value"] for m in meta["methods"]] == ["hybrid", "stats", "semantic"]
    assert meta["defaults"]["min_games"] == 20
    assert meta["llm"] == {"available": False, "status": "LLM disabled (FULLCOURT_LLM_PROVIDER=none)"}
    assert meta["index_size"] == 4


def test_search_returns_display_ready_results(client):
    body = client.post("/api/search", json={"question": "pass first playmakers who drive and kick"}).json()
    assert body["summary"]["method"] == "hybrid"
    assert body["summary"]["parser"] == "rules"
    top = body["results"][0]
    assert top["player_name"] == "Fixture Guard Three"
    assert {s["label"]: s["value"] for s in top["key_stats"]}["Assists per game"] == "9.0"
    assert top["passages"][0]["document_id"] == "guard3-doc-0"
    assert top["explanation"]["source"] == "template"
    assert any(m["pool_rank"].endswith("percentile") for g in top["goals"] for m in g["metrics"])


def test_stats_only_search_has_no_passages(client):
    body = client.post("/api/search", json={"question": "good playmakers", "method": "stats"}).json()
    assert body["results"] and all(r["passages"] == [] for r in body["results"])
    assert body["summary"]["eligibility_applied"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "playmakers", "method": "vibes"},
        {"question": "playmakers", "stat_weight": 1.5},
        {"question": "playmakers", "missing_semantic_policy": "maybe"},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/api/search", json=payload).status_code == 422


def test_whitespace_question_gets_a_clear_error(client):
    response = client.post("/api/search", json={"question": "   "})
    assert response.status_code == 422
    assert response.json()["detail"] == "Please enter a question."


def test_score_contributions_add_up_to_the_final_score(client):
    body = client.post("/api/search", json={"question": "pass first playmakers who drive and kick", "stat_weight": 0.7}).json()
    for player in body["results"]:
        score = player["score"]
        assert score["stat_contribution"] + score["style_contribution"] == pytest.approx(score["final"])


def test_stats_source_is_returned_as_labelled_links():
    url = (
        "https://www.basketball-reference.com/leagues/NBA_2026_per_game.html ; "
        "https://www.basketball-reference.com/leagues/NBA_2026_advanced.html ; "
        "https://www.basketball-reference.com/leagues/NBA_2026_play-by-play.html"
    )
    assert api._stats_tables(url) == [
        {"label": "per game", "url": "https://www.basketball-reference.com/leagues/NBA_2026_per_game.html"},
        {"label": "advanced", "url": "https://www.basketball-reference.com/leagues/NBA_2026_advanced.html"},
        {"label": "play-by-play", "url": "https://www.basketball-reference.com/leagues/NBA_2026_play-by-play.html"},
    ]
    assert api._stats_tables("fixture://") == [{"label": "source", "url": "fixture://"}]
