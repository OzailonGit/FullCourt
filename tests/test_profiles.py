import re

import pytest

from src.stats.profiles import PoolPercentiles, build_profile_text, build_search_text, style_tags

POOL = [
    {"dunk_share": 0.01 * i, "assisted_share_2pt": 0.01 * i, "usage_rate": 0.30 - 0.002 * i, "three_point_attempt_rate": 0.5 - 0.005 * i}
    for i in range(100)
]
COLUMNS = ["dunk_share", "assisted_share_2pt", "usage_rate", "three_point_attempt_rate"]

RIM_RUNNER = {
    "player_name": "Fixture Rim Runner",
    "season": "2025-26",
    "minutes_share_c": 0.9,
    "minutes_share_pf": 0.1,
    "minutes_per_game": 21.0,
    "games_played": 68,
    "points_per_game": 6.5,
    "dunk_share": 0.95,
    "assisted_share_2pt": 0.90,
    "usage_rate": 0.11,
    "three_point_attempt_rate": 0.0,
    "assisted_share_3pt": None,
}


def test_pool_percentiles_use_midpoint_for_ties():
    percentiles = PoolPercentiles([{"x": v} for v in (1, 2, 2, 3)], ["x"])
    assert percentiles("x", 2) == pytest.approx(0.5)
    assert percentiles("x", 3) == pytest.approx(1.0)
    assert percentiles("x", None) is None


def test_style_tags_follow_percentile_rules():
    tags = style_tags(RIM_RUNNER, PoolPercentiles(POOL, COLUMNS))
    assert "rim-running finisher and lob threat who scores off teammates' passes" in tags
    assert "low-usage role player" in tags
    assert "rarely shoots threes" in tags
    assert not any("primary scorer" in t for t in tags)


def test_profile_restates_numbers_and_skips_missing_values():
    text = build_profile_text(RIM_RUNNER, PoolPercentiles(POOL, COLUMNS), eligible=True)
    assert "plays mostly center (90% of minutes)" in text
    assert "95% dunks" in text
    assert "90% of made twos assisted" in text
    assert "made threes assisted" not in text  # NULL value, so the clause is left out
    assert "Style tags from percentile rules:" in text


def test_small_samples_get_no_tags():
    text = build_profile_text(RIM_RUNNER, PoolPercentiles(POOL, COLUMNS), eligible=False)
    assert "Small sample: no comparison with rotation players and no style tags." in text
    assert "lob threat" not in text


def test_profile_text_uses_no_pronouns():
    text = build_profile_text(RIM_RUNNER, PoolPercentiles(POOL, COLUMNS), eligible=True)
    assert not re.search(r"\b(he|his|him|she|her|hers)\b", text, re.I)


def test_profile_text_leaves_out_the_player_name():
    text = build_profile_text(RIM_RUNNER, PoolPercentiles(POOL, COLUMNS), eligible=True)
    assert "Fixture" not in text


def test_search_text_has_no_numbers_but_keeps_role_and_comparisons():
    text = build_search_text(RIM_RUNNER, PoolPercentiles(POOL, COLUMNS), eligible=True)
    assert not re.search(r"\d", text)
    assert text.startswith("Role: center, big man.")
    assert "very high: dunk rate, assisted twos" in text
    assert "very low: usage, three-point volume" in text
    assert "lob threat" in text


def test_profile_includes_the_same_comparison_as_the_search_text():
    percentiles = PoolPercentiles(POOL, COLUMNS)
    profile = build_profile_text(RIM_RUNNER, percentiles, eligible=True)
    assert "Compared with rotation players: very high: dunk rate, assisted twos" in profile


def test_small_sample_search_text_has_role_only():
    text = build_search_text(RIM_RUNNER, PoolPercentiles(POOL, COLUMNS), eligible=False)
    assert text == "Role: center, big man. Small sample: no comparison with rotation players."
