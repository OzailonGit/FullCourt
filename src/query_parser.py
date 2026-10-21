"""Turn a natural-language question into a validated SearchPlan.

parse_question() uses the LLM when one is configured and falls back to the deterministic
rule-based parser when there is no LLM or the LLM call fails. Either way the result is validated
by the same Pydantic models before anything touches the database.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from src.llm.client import LLMClient, LLMError
from src.llm.prompts import PARSER_SYSTEM
from src.schemas import Goal, HardFilter, LLMSearchPlan, Position, Preference, SearchPlan


@dataclass
class ParseResult:
    plan: SearchPlan
    warnings: list[str] = field(default_factory=list)



POSITION_PATTERNS: list[tuple[str, list[Position]]] = [
    (r"\bpoint guards?\b", [Position.PG]),
    (r"\bshooting guards?\b", [Position.SG]),
    (r"\bsmall forwards?\b", [Position.SF]),
    (r"\bpower forwards?\b", [Position.PF]),
    (r"\b(?<!point )(?<!shooting )guards?\b|\bbackcourt\b", [Position.PG, Position.SG]),
    (r"\bwings?\b", [Position.SG, Position.SF]),
    (r"\b(?<!small )(?<!power )forwards?\b", [Position.SF, Position.PF]),
    (r"\bcenters?\b|\bcentres?\b", [Position.C]),
    (r"\bbigs\b|\bbig men\b|\bbig man\b|\bfrontcourt\b", [Position.PF, Position.C]),
]


STAT_PHRASES: list[tuple[str, str]] = [
    (r"true shooting(?: percentage)?|ts ?%?", "true_shooting_pct"),
    (r"(?:from three|from 3|3-?point percentage|three-?point percentage|3p ?%)", "three_point_pct"),
    (r"(?:threes|3s|3-?pointers|three-?pointers|3pa|three-?point attempts)(?: attempted)?", "three_point_attempts_per_game"),
    (r"free[- ]throw(?: percentage)?|ft ?%", "free_throw_pct"),
    (r"field[- ]goal(?: percentage)?|fg ?%", "field_goal_pct"),
    (r"usage(?: rate)?", "usage_rate"),
    (r"points|ppg|pts", "points_per_game"),
    (r"assists|apg|ast", "assists_per_game"),
    (r"rebounds|rpg|boards", "rebounds_per_game"),
    (r"steals|spg", "steals_per_game"),
    (r"blocks|bpg", "blocks_per_game"),
    (r"turnovers|tov", "turnovers_per_game"),
    (r"minutes|mpg", "minutes_per_game"),
    (r"games(?: played)?", "games_played"),
]
STAT_ALTERNATION = "|".join(f"(?:{p})" for p, _ in STAT_PHRASES)
MIN_WORDS = r"at least|no fewer than|no less than|minimum of|a minimum of|min(?:imum)?|over|more than|above|greater than"
MAX_WORDS = r"at most|no more than|fewer than|less than|under|below|maximum of|max(?:imum)?"
FILTER_PATTERN = re.compile(
    rf"(?:(?P<op>{MIN_WORDS}|{MAX_WORDS})\s+)?(?P<num>\d+(?:\.\d+)?)(?P<plus>\+)?\s*(?P<pct>%|percent)?\s*"
    rf"(?:(?:per game|a game|per contest)\s+)?(?P<stat>{STAT_ALTERNATION})\b",
    re.I,
)
CONNECTOR = re.compile(r"^\s*(?:,|and|,\s*and|&)\s*$", re.I)


PREFERENCE_RULES: list[tuple[str, Goal, str]] = [
    # direction "low"/"moderate" rules come first so the generic "high" rule for the same goal is skipped
    (r"without scoring (?:a lot|much)|don'?t score (?:a lot|much)|low[- ]scoring|non-?scorers?|low[- ]usage|not (?:big|high|volume) scorers?", Goal.scoring_volume, "low"),
    (r"(?:average|decent|moderate|okay|ok|solid) (?:scorers?|scoring)", Goal.scoring_volume, "moderate"),
    (r"(?:average|decent|moderate|okay|ok) (?:playmakers?|playmaking|passers?|passing)", Goal.playmaking, "moderate"),
    (r"efficien|true shooting|ts%", Goal.scoring_efficiency, "high"),
    (r"\bscor(?:er|ers|ing)\b|\bbuckets?\b|\bpoints\b", Goal.scoring_volume, "high"),
    (r"playmak|\bpass(?:er|ers|ing)\b|assists?\b|set(?:s|ting)? up teammates|create (?:opportunities|shots|looks) for (?:teammates|others)|create for (?:teammates|others)|facilitat|distribut|table[- ]setter", Goal.playmaking, "high"),
    (r"create (?:their|his) own (?:shot|shots|offense)|self[- ]creat|shot[- ]creat|off the dribble|isolation|\biso\b|pull[- ]?ups?|create advantages|advantage creat|on[- ]ball creat", Goal.shot_creation, "high"),
    (r"take care of the ball|ball security|low[- ]turnover|few turnovers|protect the ball|don'?t turn (?:it|the ball) over", Goal.ball_security, "high"),
    (r"three[- ]point shoot|3[- ]point shoot|\bshooters?\b|shooting from (?:three|deep|range)|from deep|floor spac|spac(?:e|es|ing) the floor|\bspacing\b|stretch (?:big|four|five)|knock down (?:threes|shots)", Goal.three_point_shooting, "high"),
    (r"rebound|crash(?:es|ing)? the glass|on the glass|\bboards\b", Goal.rebounding, "high"),
    (r"defen|steals|blocks|rim protect|shot[- ]block|\bstocks\b|disrupt|3[- ]and[- ]d|three[- ]and[- ]d", Goal.defense_activity, "high"),
    (r"spot[- ]?up|catch[- ]and[- ]shoot|corner (?:three|3)s?|3[- ]and[- ]d|three[- ]and[- ]d", Goal.spot_up_shooting, "high"),
    (r"attack(?:s|ing)? the (?:rim|basket)|rim pressure|\bdriv(?:e|es|er|ers|ing)\b|slash|get(?:s|ting)? to the (?:line|rim)|draw(?:s|ing)? fouls|downhill|paint touches", Goal.rim_pressure, "high"),
    (r"rim[- ]?run|\blobs?\b|roll man|rolls? to the (?:rim|basket)|vertical spacer|above the rim|dunk|finish(?:es|er|ers|ing)? at the rim", Goal.rim_finishing, "high"),
    (r"post[- ]?ups?\b|post scor|in the post|low post|back to the basket|post moves|\bpost players?\b", Goal.post_scoring, "high"),
    (r"mid[- ]?range|\belbows?\b|long twos?", Goal.midrange_scoring, "high"),
]

UNSUPPORTED_RULES: list[tuple[str, str]] = [
    (r"clutch", "clutch performance (no clutch or late-game splits in the dataset)"),
    (r"salar|contract|cap hit", "salary and contract information"),
    (r"injur|health", "injury or health history"),
    (r"playoff|postseason", "playoff performance (only regular-season statistics are loaded)"),
    (r"on[/ -]off|plus[- ]minus|net rating|lineup", "on/off or lineup impact"),
    (r"draft|prospect|college", "draft or college information"),
]

REFERENCE_PATTERN = re.compile(
    r"(?:similar (?:style |game |profile |playing style )?to|plays? like|like|comparable to|style of|in the mold of|resemble|reminiscent of)\s+"
    r"(?P<name>[A-Z][\w'.\-]+(?:\s+(?:[A-Z][\w'.\-]+|Jr\.?|Sr\.?|II|III|IV))+)"
)
SEASON_PATTERN = re.compile(r"\b(20\d{2})\s*[-–/]\s*(\d{2})\b")
FILLER_WORDS = set(
    "a an and are as at be best by can do find for from get give good great in is me of on or per game games "
    "player players season show some that the their them they this to top what which who whose with "
    "list rank ranked nba league guys".split()
)


def parse_rules(question: str, default_season: str) -> ParseResult:
    warnings: list[str] = []

    season_match = SEASON_PATTERN.search(question)
    season = f"{season_match.group(1)}-{season_match.group(2)}" if season_match else default_season

    lowered = question.lower()
    positions: list[Position] = []
    for pattern, matched in POSITION_PATTERNS:
        if re.search(pattern, lowered):
            positions.extend(p for p in matched if p not in positions)

    hard_filters, filter_spans = _extract_filters(question)

    # Remove filter spans so "at least 20 points" does not also become a "scorer" preference.
    text = question
    for start, end in sorted(filter_spans, reverse=True):
        text = text[:start] + " " + text[end:]
    text_lower = text.lower()

    preferences: list[Preference] = []
    for pattern, goal, direction in PREFERENCE_RULES:
        if any(p.goal == goal for p in preferences):
            continue
        if re.search(pattern, text_lower):
            preferences.append(Preference(goal=goal, direction=direction))

    unsupported = [message for pattern, message in UNSUPPORTED_RULES if re.search(pattern, lowered)]

    reference = REFERENCE_PATTERN.search(question)
    reference_player = reference.group("name").rstrip(".?!,") if reference else None
    if reference_player is None and re.search(r"similar|comparable|in the mold of", lowered):
        warnings.append("The question asks for similar players but does not name a reference player.")

    # Only run semantic search when something descriptive is left after removing filters,
    # positions and filler words; otherwise vector search would rank on noise.
    content_words = [
        w for w in re.findall(r"[a-z']+", SEASON_PATTERN.sub(" ", text_lower))
        if w not in FILLER_WORDS and not any(re.fullmatch(p, w) for p, _ in POSITION_PATTERNS)
    ]
    semantic_query = " ".join(text.split()) if (content_words or reference_player) else ""
    plan = SearchPlan(
        season=season,
        season_was_defaulted=season_match is None,
        positions=positions,
        hard_filters=hard_filters,
        preferences=preferences,
        semantic_query=semantic_query,
        reference_player=reference_player,
        unsupported_requests=unsupported,
        parser="rules",
    )
    return ParseResult(plan, warnings)


def _extract_filters(question: str) -> tuple[list[HardFilter], list[tuple[int, int]]]:
    filters: list[HardFilter] = []
    spans: list[tuple[int, int]] = []
    previous_op: str | None = None
    previous_end = -1
    for match in FILTER_PATTERN.finditer(question):
        stat_text = match.group("stat").lower()
        metric = next(name for pattern, name in STAT_PHRASES if re.fullmatch(pattern, stat_text, re.I))
        op_words = (match.group("op") or "").lower()
        if op_words:
            op = "<=" if re.fullmatch(MAX_WORDS, op_words) else ">="
        elif match.group("plus"):
            op = ">="
        elif previous_op and CONNECTOR.match(question[previous_end:match.start()]):
            op = previous_op  # "at least 20 points and 5 assists"
        else:
            continue  # a bare number without a comparison is too ambiguous to enforce
        value = float(match.group("num"))
        try:
            filters.append(HardFilter(metric=metric, op=op, value=value))
        except ValidationError:
            continue
        spans.append(match.span())
        previous_op, previous_end = op, match.end()
    return filters, spans


def parse_llm(question: str, default_season: str, llm: LLMClient) -> ParseResult:
    raw = llm.parse(PARSER_SYSTEM, question, LLMSearchPlan, effort="low")
    plan = SearchPlan(
        season=raw.season or default_season,
        season_was_defaulted=raw.season is None,
        positions=raw.positions,
        hard_filters=raw.hard_filters,
        preferences=_dedupe_preferences(raw.preferences),
        semantic_query=raw.semantic_query.strip() or question,
        reference_player=raw.reference_player,
        unsupported_requests=raw.unsupported_requests,
        parser="llm",
    )
    return ParseResult(plan)


def _dedupe_preferences(preferences: list[Preference]) -> list[Preference]:
    seen: dict[Goal, Preference] = {}
    for p in preferences:
        seen.setdefault(p.goal, p)
    return list(seen.values())


def parse_question(question: str, default_season: str, llm: LLMClient | None = None) -> ParseResult:
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")
    if llm is not None:
        try:
            return parse_llm(question, default_season, llm)
        except (LLMError, ValidationError, ValueError) as e:
            result = parse_rules(question, default_season)
            result.warnings.insert(0, f"LLM parsing failed ({e}); used the rule-based parser instead.")
            return result
    return parse_rules(question, default_season)
