"""Write a plain-language style profile for every player from their own statistics.

Every player gets two texts:
  * the profile (build_profile_text), shown in the UI and given to the LLM, which restates the
    numbers and ends with how the player compares with rotation players and their style tags;
  * the search summary (build_search_text), which is what gets embedded. It keeps only the role,
    the comparisons and the tags, with no numbers.

The split exists because the embedding model barely reads numbers. When the full profile was
embedded, the shared template wording dominated and every profile looked almost identical (cosine
0.98 to 0.997 between players), so style search ranked on noise.

None of this is scouting text. The comparisons and tags come from fixed percentile rules applied
against rotation players (the default eligibility pool), and are only added for players in that
pool, because rate stats from a handful of games would produce misleading labels. The text avoids
pronouns so it never guesses anyone's gender.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass

DESCRIPTION_TYPE = "stat-based profile"

POSITION_LABELS = {
    "minutes_share_pg": "point guard",
    "minutes_share_sg": "shooting guard",
    "minutes_share_sf": "small forward",
    "minutes_share_pf": "power forward",
    "minutes_share_c": "center",
}
# Everyday role words, so questions about "guards", "wings" or "bigs" match the search summary.
ROLE_WORDS = {
    "minutes_share_pg": ["point guard", "guard"],
    "minutes_share_sg": ["shooting guard", "guard", "wing"],
    "minutes_share_sf": ["small forward", "wing", "forward"],
    "minutes_share_pf": ["power forward", "forward", "big"],
    "minutes_share_c": ["center", "big man"],
}
MAIN_POSITION_SHARE = 0.2

# Traits compared against rotation players, as (column, wording).
TRAITS = [
    ("usage_rate", "usage"),
    ("points_per_game", "scoring volume"),
    ("true_shooting_pct", "scoring efficiency"),
    ("shot_share_0_3ft", "share of shots at the rim"),
    ("dunk_share", "dunk rate"),
    ("shot_share_3_10ft", "short-range shots (floaters, hooks, post-ups)"),
    ("midrange_share", "mid-range shots"),
    ("three_point_attempt_rate", "three-point volume"),
    ("corner_three_share", "corner threes"),
    ("assisted_share_2pt", "assisted twos"),
    ("assisted_share_3pt", "assisted threes"),
    ("shooting_fouls_drawn_per_game", "fouls drawn"),
    ("assist_rate", "playmaking"),
    ("rebound_rate", "rebounding"),
    ("block_rate", "shot blocking"),
    ("steal_rate", "steals"),
]
# Percentile cut-offs for the comparison wording; traits in between are average and left out.
TRAIT_BUCKETS = [("very high", 0.9, 1.0), ("high", 0.7, 0.9), ("low", 0.1, 0.3), ("very low", 0.0, 0.1)]


class PoolPercentiles:
    """Percentile of any value against the eligible pool's distribution for each column."""

    def __init__(self, pool: list[dict], columns: list[str]):
        self.sorted_values = {c: sorted(r[c] for r in pool if r.get(c) is not None) for c in columns}

    def __call__(self, column: str, value: float | None) -> float | None:
        values = self.sorted_values.get(column) or []
        if value is None or len(values) < 2:
            return None
        # Midpoint of the tied range, matching the average-rank convention used in ranking.py.
        rank = (bisect_left(values, value) + bisect_right(values, value) - 1) / 2
        return min(max(rank / (len(values) - 1), 0.0), 1.0)


@dataclass(frozen=True)
class TagRule:
    tag: str
    applies: Callable[[Callable[[str], float | None], dict], bool]


def _at_least(p: Callable[[str], float | None], column: str, threshold: float) -> bool:
    value = p(column)
    return value is not None and value >= threshold


def _at_most(p: Callable[[str], float | None], column: str, threshold: float) -> bool:
    value = p(column)
    return value is not None and value <= threshold


def _plays_big(row: dict) -> bool:
    return (row.get("minutes_share_pf") or 0) + (row.get("minutes_share_c") or 0) >= 0.5


# Thresholds are percentiles within the eligible pool. They are judgment calls meant to read
# like common basketball shorthand, and are listed here so they can be reviewed and tuned.
STYLE_TAG_RULES: list[TagRule] = [
    TagRule("high-usage primary scorer", lambda p, r: _at_least(p, "usage_rate", 0.85)),
    TagRule("low-usage role player", lambda p, r: _at_most(p, "usage_rate", 0.2)),
    TagRule("efficient scorer", lambda p, r: _at_least(p, "true_shooting_pct", 0.8)),
    TagRule(
        "creates own shot off the dribble, with most baskets unassisted",
        lambda p, r: _at_most(p, "assisted_share_2pt", 0.2) and _at_least(p, "usage_rate", 0.6),
    ),
    TagRule("primary playmaker who sets up teammates", lambda p, r: _at_least(p, "assist_rate", 0.85)),
    TagRule(
        "secondary playmaker and capable passer",
        lambda p, r: _at_least(p, "assist_rate", 0.65) and not _at_least(p, "assist_rate", 0.85),
    ),
    TagRule(
        "attacks the rim, drives downhill and draws fouls",
        lambda p, r: _at_least(p, "shot_share_0_3ft", 0.7) and _at_least(p, "shooting_fouls_drawn_per_game", 0.75),
    ),
    TagRule(
        "rim-running finisher and lob threat who scores off teammates' passes",
        lambda p, r: _at_least(p, "dunk_share", 0.85) and _at_least(p, "assisted_share_2pt", 0.6),
    ),
    TagRule(
        "catch-and-shoot, spot-up three-point shooter",
        lambda p, r: _at_least(p, "assisted_share_3pt", 0.6) and _at_least(p, "three_point_attempt_rate", 0.7),
    ),
    TagRule("corner three specialist", lambda p, r: _at_least(p, "corner_three_share", 0.85)),
    TagRule(
        "stretch big who spaces the floor with threes",
        lambda p, r: _plays_big(r) and _at_least(p, "three_point_attempt_rate", 0.6),
    ),
    TagRule("rarely shoots threes", lambda p, r: _at_most(p, "three_point_attempt_rate", 0.1)),
    TagRule(
        "mid-range scorer",
        lambda p, r: _at_least(p, "midrange_share", 0.85),
    ),
    TagRule(
        "post and short-range scorer (hooks, floaters, post-ups)",
        lambda p, r: _at_least(p, "shot_share_3_10ft", 0.85),
    ),
    TagRule("rim protector who blocks shots", lambda p, r: _at_least(p, "block_rate", 0.85)),
    TagRule("active hands on defense, generates steals", lambda p, r: _at_least(p, "steal_rate", 0.85)),
    TagRule("draws charges", lambda p, r: _at_least(p, "charges_drawn_per_game", 0.9)),
    TagRule("strong rebounder", lambda p, r: _at_least(p, "rebound_rate", 0.85)),
    TagRule(
        "3-and-D role player",
        lambda p, r: _at_least(p, "three_point_attempt_rate", 0.6)
        and _at_least(p, "three_point_pct", 0.5)
        and (_at_least(p, "steal_rate", 0.6) or _at_least(p, "block_rate", 0.6))
        and _at_most(p, "usage_rate", 0.5),
    ),
]

TAG_COLUMNS = [
    "usage_rate", "true_shooting_pct", "assisted_share_2pt", "assisted_share_3pt", "assist_rate",
    "shot_share_0_3ft", "shooting_fouls_drawn_per_game", "dunk_share", "three_point_attempt_rate",
    "corner_three_share", "midrange_share", "shot_share_3_10ft", "block_rate", "steal_rate",
    "charges_drawn_per_game", "rebound_rate", "three_point_pct",
]
PERCENTILE_COLUMNS = sorted(set(TAG_COLUMNS) | {column for column, _ in TRAITS})


def with_derived_columns(row: dict) -> dict:
    """Add columns the profile needs that are sums of stored columns."""
    short, long = row.get("shot_share_10_16ft"), row.get("shot_share_16ft_to_3pt")
    return {**row, "midrange_share": None if short is None or long is None else short + long}


def style_tags(row: dict, percentiles: PoolPercentiles) -> list[str]:
    def p(column: str) -> float | None:
        return percentiles(column, row.get(column))

    return [rule.tag for rule in STYLE_TAG_RULES if rule.applies(p, row)]


def trait_comparison(row: dict, percentiles: PoolPercentiles) -> dict[str, list[str]]:
    """Group the player's non-average traits by how they compare with rotation players."""
    row = with_derived_columns(row)
    buckets: dict[str, list[str]] = {name: [] for name, _, _ in TRAIT_BUCKETS}
    for column, wording in TRAITS:
        p = percentiles(column, row.get(column))
        if p is None:
            continue
        for name, low, high in TRAIT_BUCKETS:
            if low <= p < high or (high == 1.0 and p == 1.0):
                buckets[name].append(wording)
                break
    return {name: traits for name, traits in buckets.items() if traits}


def _comparison_sentence(row: dict, percentiles: PoolPercentiles) -> str:
    comparison = trait_comparison(row, percentiles)
    if not comparison:
        return "Compared with rotation players: no trait stands out."
    return "Compared with rotation players: " + "; ".join(f"{name}: {', '.join(t)}" for name, t in comparison.items()) + "."


def _tags_sentence(row: dict, percentiles: PoolPercentiles) -> str:
    tags = style_tags(with_derived_columns(row), percentiles)
    return f"Style tags from percentile rules: {'; '.join(tags)}." if tags else "Style tags from percentile rules: none stand out."


def build_search_text(row: dict, percentiles: PoolPercentiles, eligible: bool) -> str:
    """Return the number-free summary that gets embedded (see the module docstring for why)."""
    words: list[str] = []
    for column, labels in ROLE_WORDS.items():
        if (row.get(column) or 0.0) >= MAIN_POSITION_SHARE:
            words += [w for w in labels if w not in words]
    role = f"Role: {', '.join(words)}." if words else "Role: position not recorded."
    if not eligible:
        return f"{role} Small sample: no comparison with rotation players."
    return " ".join([role, _comparison_sentence(row, percentiles), _tags_sentence(row, percentiles)])


def _pct(value: float | None) -> str | None:
    return None if value is None else f"{value * 100:.0f}%"


def _num(value: float | None, digits: int = 1) -> str | None:
    return None if value is None else f"{value:.{digits}f}"


def _sentence(label: str, clauses: list[str | None]) -> str | None:
    present = [c for c in clauses if c]
    return f"{label}: {', '.join(present)}." if present else None


def _clause(template: str, *values: str | None) -> str | None:
    return None if any(v is None for v in values) else template.format(*values)


def build_profile_text(row: dict, percentiles: PoolPercentiles, eligible: bool) -> str:
    """Return the profile text for one player-season row."""
    row = with_derived_columns(row)
    positions = sorted(
        ((row.get(column) or 0.0, label) for column, label in POSITION_LABELS.items()), reverse=True
    )
    main_positions = [
        f"{label} ({share * 100:.0f}% of minutes)" for share, label in positions if share >= MAIN_POSITION_SHARE
    ]

    sentences = [
        # The player's name is left out on purpose: in the embedding it would make "similar to
        # Nikola Jokić" match other players named Nikola instead of similar playing styles.
        f"Statistical style profile for {row['season']}, written from Basketball-Reference numbers.",
        _sentence("Role", [
            _clause("plays mostly {}", " and ".join(main_positions) if main_positions else None),
            _clause("{} minutes per game over {} games", _num(row.get("minutes_per_game")), _num(row.get("games_played"), 0)),
        ]),
        _sentence("Scoring", [
            _clause("{} points per game", _num(row.get("points_per_game"))),
            _clause("{} true shooting", _pct(row.get("true_shooting_pct"))),
            _clause("usage rate {}", _pct(row.get("usage_rate"))),
        ]),
        _sentence("Shot diet", [
            _clause("{} of shots at the rim (0-3 ft)", _pct(row.get("shot_share_0_3ft"))),
            _clause("{} from 3-10 ft", _pct(row.get("shot_share_3_10ft"))),
            _clause("{} mid-range (10 ft to the arc)", _pct(row.get("midrange_share"))),
            _clause("{} threes", _pct(row.get("three_point_attempt_rate"))),
            _clause("{} dunks", _pct(row.get("dunk_share"))),
            _clause("average shot distance {} ft", _num(row.get("average_shot_distance"))),
        ]),
        _sentence("Finishing", [
            _clause("{} at the rim", _pct(row.get("fg_pct_0_3ft"))),
            _clause("{} from 3-10 ft", _pct(row.get("fg_pct_3_10ft"))),
            _clause("{} from mid-range (10-16 ft)", _pct(row.get("fg_pct_10_16ft"))),
        ]),
        _sentence("Shot creation", [
            _clause("{} of made twos assisted", _pct(row.get("assisted_share_2pt"))),
            _clause("{} of made threes assisted", _pct(row.get("assisted_share_3pt"))),
        ]),
        _sentence("Three-point shooting", [
            _clause("{} attempts per game", _num(row.get("three_point_attempts_per_game"))),
            _clause("{} made", _pct(row.get("three_point_pct"))),
            _clause("{} of attempts from the corners", _pct(row.get("corner_three_share"))),
        ]),
        _sentence("Getting to the line", [
            _clause("draws {} shooting fouls per game", _num(row.get("shooting_fouls_drawn_per_game"))),
            _clause("{} and-ones per game", _num(row.get("and_ones_per_game"))),
        ]),
        _sentence("Passing", [
            _clause("{} assists per game", _num(row.get("assists_per_game"))),
            _clause("assist rate {}", _pct(row.get("assist_rate"))),
            _clause("{} points created by assists per game", _num(row.get("points_created_by_assists_per_game"))),
            _clause("turnover rate {}", _pct(row.get("turnover_rate"))),
        ]),
        _sentence("Rebounding and defense", [
            _clause("rebound rate {}", _pct(row.get("rebound_rate"))),
            _clause("{} steals per game", _num(row.get("steals_per_game"))),
            _clause("{} blocks per game", _num(row.get("blocks_per_game"))),
            _clause("{} charges drawn per game", _num(row.get("charges_drawn_per_game"), 2)),
        ]),
    ]

    if eligible:
        sentences += [_comparison_sentence(row, percentiles), _tags_sentence(row, percentiles)]
    else:
        sentences.append("Small sample: no comparison with rotation players and no style tags.")
    return " ".join(s for s in sentences if s)
