"""Streamlit UI for the NBA research assistant.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from src import pipeline
from src.generation.explanations import goal_label, percentile_text
from src.retrieval.hybrid import format_stat
from src.schemas import METRICS
from src.stats.queries import available_seasons

METHOD_LABELS = {"hybrid": "Hybrid (statistics + descriptions)", "stats": "Statistics only", "semantic": "Descriptions only (semantic)"}
EXAMPLES = [
    "Find efficient scoring guards who are good playmakers and can create their own shots.",
    "Which players are efficient scorers who also create opportunities for teammates?",
    "Find playmakers who create advantages without scoring a lot.",
    "Bigs who shoot at least 35% from three and grab 8+ rebounds",
    "Find players with a similar style to Jalen Brunson",
]
KEY_METRICS = ["points_per_game", "assists_per_game", "true_shooting_pct", "usage_rate", "assist_rate", "minutes_per_game"]

st.set_page_config(page_title="FullCourt: NBA research assistant", page_icon="🏀", layout="wide")


@st.cache_resource(show_spinner="Loading database and description index…")
def get_engine():
    return pipeline.load_engine()


@st.cache_resource
def get_llm():
    return pipeline.load_llm()


try:
    engine = get_engine()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
llm, llm_status = get_llm()

with st.sidebar:
    st.header("Search settings")
    method = st.radio("Retrieval method", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
    seasons = available_seasons(engine.conn)
    season = st.selectbox("Season", seasons, help="Used when the question does not name a season.")
    use_eligibility = st.checkbox(
        f"Default eligibility (≥ {config.DEFAULT_MIN_GAMES} games, ≥ {config.DEFAULT_MIN_MINUTES:g} MPG)",
        value=True,
        help="A sample-size floor so a player with two games does not top a rate stat. "
        "This is a default, not something inferred from your question. The semantic-only baseline ignores it.",
    )
    stat_weight = st.slider("Statistics weight (hybrid)", 0.0, 1.0, config.STAT_WEIGHT, 0.05)
    missing_policy = st.selectbox(
        "Players without a style description",
        ["zero", "neutral"],
        index=["zero", "neutral"].index(config.MISSING_SEMANTIC_POLICY),
        format_func={"zero": "No semantic credit", "neutral": "Median semantic score"}.get,
    )
    st.divider()
    st.caption(llm_status)
    if llm is None:
        st.caption("Using the rule-based parser and template explanations.")
    st.caption(
        f"Index: {len(engine.index)} description passages" if engine.index else "Index: not built (semantic search unavailable)"
    )

st.title("FullCourt")
st.caption("Ask about NBA players' statistics and playing style. Every number comes from the loaded dataset; "
           "every style claim links to its source passage.")

if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0]
with st.form("search"):
    question = st.text_input("Question", key="question")
    submitted = st.form_submit_button("Search", type="primary")
with st.expander("Example questions"):
    for example in EXAMPLES:
        st.code(example, language=None)

if not submitted:
    st.stop()
if not question.strip():
    st.warning("Please enter a question.")
    st.stop()

settings = pipeline.default_settings(use_eligibility)
settings.stat_weight, settings.semantic_weight = stat_weight, 1.0 - stat_weight
settings.missing_semantic_policy = missing_policy

with st.spinner("Searching…"):
    try:
        output = pipeline.run(question, engine, method, settings, llm, season=season)
    except ValueError as e:
        st.error(str(e))
        st.stop()
result = output.result
plan = result.plan

summary = [f"**Method:** {METHOD_LABELS[method]}", f"**Season:** {plan.season}" + (" (default)" if plan.season_was_defaulted else "")]
summary.append(f"**Parser:** {'LLM' if plan.parser == 'llm' else 'rule-based'}")
summary.append(f"**Candidates after filters:** {result.candidate_count}")
st.markdown(" · ".join(summary))

for message in result.messages + output.warnings:
    st.info(message)
for item in plan.unsupported_requests:
    st.warning(f"Not supported by this dataset: {item}")

with st.expander("Search plan and timings"):
    left, right = st.columns([2, 1])
    left.json(plan.model_dump(mode="json"))
    right.dataframe(
        pd.DataFrame({"stage": list(result.timings_ms), "ms": [round(v, 1) for v in result.timings_ms.values()]}),
        hide_index=True,
    )
    if method != "semantic" and use_eligibility:
        right.caption(f"Default eligibility applied: ≥ {config.DEFAULT_MIN_GAMES} games and ≥ {config.DEFAULT_MIN_MINUTES:g} MPG.")

if not result.results:
    st.warning("No results. Try removing a filter, choosing another method, or turning off default eligibility.")
    st.stop()

explanations = {e.player_id: e for e in output.explanations}
for player in result.results:
    b = player.breakdown
    stats = player.stats
    with st.container(border=True):
        head, score = st.columns([4, 1])
        team = stats["team"] + (f" ({stats['teams']})" if stats.get("teams") and stats["team"] != stats["teams"] else "")
        head.subheader(f"{player.rank}. {player.player_name}")
        head.caption(f"{stats.get('position') or 'Position not listed'} · {team} · {stats['season']} · "
                     f"{format_stat('games_played', stats['games_played'])} games")
        score.metric("Score", f"{b.final_score:.2f}")

        cols = st.columns(len(KEY_METRICS))
        for col, metric in zip(cols, KEY_METRICS):
            col.metric(METRICS[metric].label, format_stat(metric, stats.get(METRICS[metric].column)))

        explanation = explanations.get(player.player_id)
        if explanation:
            label = "Generated explanation (grounding-checked)" if explanation.source == "llm" else "Explanation (template)"
            st.markdown(f"**{label}:** {explanation.text}")

        breakdown_rows = [
            {"component": "Statistical fit", "score": b.stat_fit},
            {"component": "Semantic relevance" + (" (no description: policy applied)" if b.semantic_missing and method == "hybrid" else ""),
             "score": b.semantic_score},
        ]
        with st.expander("Score breakdown"):
            st.caption("Components used: " + (", ".join(b.components_used) or "none"))
            st.dataframe(pd.DataFrame(breakdown_rows), hide_index=True)
            goal_rows = []
            for g in b.goals:
                goal_score = None if g.score is None else round(g.score, 3)
                if not g.metrics:  # e.g. profile similarity to a reference player
                    goal_rows.append({"goal": g.goal, "goal score": goal_score, "metric": "", "value": "", "pool rank": ""})
                for m in g.metrics:
                    goal_rows.append({
                        "goal": goal_label(g.goal, g.direction),
                        "goal score": goal_score,
                        "metric": METRICS[m.metric].label,
                        "value": format_stat(m.metric, m.value),
                        "pool rank": percentile_text(m.percentile, m.lower_is_better),
                    })
            if goal_rows:
                st.dataframe(pd.DataFrame(goal_rows), hide_index=True)
            for note in {g.proxy_note for g in b.goals if g.proxy_note}:
                st.caption(f"Proxy: {note}")
            st.caption(f"Statistics source: {stats['source_name']} ({stats['source_url']})")

        with st.expander(f"Style evidence ({len(player.passages)} passage{'s' if len(player.passages) != 1 else ''})"):
            if method == "stats":
                st.caption("The statistics-only method does not use style descriptions.")
            elif not player.passages:
                st.caption("No description of this player is available in the corpus.")
            for passage in player.passages:
                st.markdown(f"> {passage.text}")
                revised = f", revision of {passage.published_at[:10]}" if passage.published_at else ""
                st.caption(
                    f"[{passage.source_name} — {passage.source_section}]({passage.source_url}){revised} · "
                    f"{passage.description_type} · {passage.license} · similarity {passage.similarity:.2f} · `{passage.document_id}`"
                )

st.caption("Statistics: Basketball-Reference. Style descriptions: Wikipedia player-profile sections (CC BY-SA 4.0). "
           "Descriptions summarize a player's career and may not reflect the selected season.")
