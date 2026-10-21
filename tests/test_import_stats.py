import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_stats import PLAY_BY_PLAY_COLUMNS, parse_rows  # noqa: E402


def test_import_converts_percent_points_and_season_totals():
    html = """<table id="pbp_stats"><tbody><tr>
      <td data-stat="name_display" data-append-csv="fixture01"><a>Fixture Player</a></td>
      <td data-stat="team_name_abbr">TST</td><td data-stat="pos">C</td><td data-stat="games">50</td>
      <td data-stat="pct_5">83</td><td data-stat="drawn_shooting">100</td><td data-stat="and1s"></td>
    </tr></tbody></table>"""
    row = parse_rows(BeautifulSoup(html, "lxml").find("table"), PLAY_BY_PLAY_COLUMNS)[0]
    assert row["minutes_share_c"] == pytest.approx(0.83)
    assert row["shooting_fouls_drawn_per_game"] == pytest.approx(2.0)
    assert row["and_ones_per_game"] is None
