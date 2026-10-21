from src.generation.explanations import (
    ExplanationBatch,
    PlayerExplanation,
    check_grounding,
    evidence_package,
    explain,
    template_explanation,
)
from src.retrieval.hybrid import RankingSettings
from src.schemas import Goal, Preference, SearchPlan
from src.stats.queries import Eligibility

QUESTION = "pass first point guards who drive and kick"


def search(engine):
    plan = SearchPlan(
        season="2025-26",
        preferences=[Preference(goal=Goal.playmaking), Preference(goal=Goal.shot_creation)],
        semantic_query=QUESTION,
    )
    return engine.search(plan, "hybrid", RankingSettings(eligibility=Eligibility(20, 10)))


class ScriptedLLM:
    name = "scripted"

    def __init__(self, texts: dict[str, tuple[str, list[str]]]):
        self.texts = texts

    def parse(self, system, user, schema, effort="low"):
        assert schema is ExplanationBatch
        return ExplanationBatch(
            players=[PlayerExplanation(player_id=pid, explanation=t, cited_document_ids=c) for pid, (t, c) in self.texts.items()]
        )


def test_template_explanations_pass_their_own_grounding_check(engine):
    result = search(engine)
    for player in result.results:
        template = template_explanation(result, player)
        package = evidence_package(result, player)
        candidate = PlayerExplanation(player_id=player.player_id, explanation=template.text, cited_document_ids=template.cited_document_ids)
        assert check_grounding(candidate, package, QUESTION) == []


def test_grounding_rejects_invented_numbers_citations_and_best_claims(engine):
    result = search(engine)
    guard3 = next(p for p in result.results if p.player_id == "guard3")
    package = evidence_package(result, guard3)

    ok = PlayerExplanation(player_id="guard3", explanation="He averages 9.0 assists per game.", cited_document_ids=["guard3-doc-0"])
    assert check_grounding(ok, package, QUESTION) == []

    invented = PlayerExplanation(player_id="guard3", explanation="He averages 11.4 assists.", cited_document_ids=[])
    assert "11.4" in check_grounding(invented, package, QUESTION)[0]

    bad_cite = PlayerExplanation(player_id="guard3", explanation="A passer.", cited_document_ids=["wing1-doc-0"])
    assert "wing1-doc-0" in check_grounding(bad_cite, package, QUESTION)[0]

    best = PlayerExplanation(player_id="guard3", explanation="He is the best passer in the league.", cited_document_ids=[])
    assert check_grounding(best, package, QUESTION)


def test_unsupported_llm_statistics_are_replaced_by_template(engine):
    result = search(engine)
    llm = ScriptedLLM({
        "guard3": ("Averages 9.0 assists; the profile describes drive-and-kick passing.", ["guard3-doc-0"]),
        "guard1": ("Averages 31.7 points per game.", []),  # not in the evidence
    })
    explanations, warnings = explain(result, QUESTION, llm)
    by_id = {e.player_id: e for e in explanations}
    assert by_id["guard3"].source == "llm"
    assert by_id["guard1"].source == "template" and by_id["guard1"].grounding_issues
    assert any("Fixture Guard One" in w for w in warnings)
    assert len(explanations) == len(result.results)  # players the LLM skipped get templates


def test_no_llm_uses_templates_and_reports_missing_style_evidence(engine):
    result = search(engine)
    explanations, warnings = explain(result, QUESTION, None)
    assert warnings == [] and all(e.source == "template" for e in explanations)
    guard2 = next(e for e in explanations if e.player_id == "guard2")
    assert "no description" in guard2.text
