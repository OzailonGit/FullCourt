"""Download Basketball-Reference season tables and load them into SQLite.

Usage:
    python scripts/import_stats.py --season 2025-26

Raw HTML is cached in data/raw/ (gitignored) so re-running does not re-download.
Basketball-Reference asks scrapers to stay under 20 requests/minute; this script makes four.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Comment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.stats.database import connect, init_schema  # noqa: E402

SOURCE_NAME = "Basketball-Reference"
USER_AGENT = "FullCourt-research-assistant/0.1 (personal portfolio project)"

# Maps each source data-stat to (column, kind). "pct_points" values are 0-100 in the source and are
# stored as fractions; "season_total" values are divided by games played and stored per game.
PER_GAME_COLUMNS = {
    "age": ("age", "int"),
    "games": ("games_played", "int"),
    "games_started": ("games_started", "int"),
    "mp_per_g": ("minutes_per_game", "float"),
    "pts_per_g": ("points_per_game", "float"),
    "ast_per_g": ("assists_per_game", "float"),
    "trb_per_g": ("rebounds_per_game", "float"),
    "stl_per_g": ("steals_per_game", "float"),
    "blk_per_g": ("blocks_per_game", "float"),
    "tov_per_g": ("turnovers_per_game", "float"),
    "fg3a_per_g": ("three_point_attempts_per_game", "float"),
    "fg_pct": ("field_goal_pct", "float"),
    "fg3_pct": ("three_point_pct", "float"),
    "ft_pct": ("free_throw_pct", "float"),
    "efg_pct": ("effective_field_goal_pct", "float"),
}
ADVANCED_COLUMNS = {
    "ts_pct": ("true_shooting_pct", "float"),
    "usg_pct": ("usage_rate", "pct_points"),
    "ast_pct": ("assist_rate", "pct_points"),
    "tov_pct": ("turnover_rate", "pct_points"),
    "trb_pct": ("rebound_rate", "pct_points"),
    "stl_pct": ("steal_rate", "pct_points"),
    "blk_pct": ("block_rate", "pct_points"),
    "bpm": ("box_plus_minus", "float"),
}
SHOOTING_COLUMNS = {
    "avg_dist": ("average_shot_distance", "float"),
    "pct_fga_00_03": ("shot_share_0_3ft", "float"),
    "pct_fga_03_10": ("shot_share_3_10ft", "float"),
    "pct_fga_10_16": ("shot_share_10_16ft", "float"),
    "pct_fga_16_xx": ("shot_share_16ft_to_3pt", "float"),
    "pct_fga_fg3a": ("three_point_attempt_rate", "float"),
    "fg_pct_00_03": ("fg_pct_0_3ft", "float"),
    "fg_pct_03_10": ("fg_pct_3_10ft", "float"),
    "fg_pct_10_16": ("fg_pct_10_16ft", "float"),
    "fg_pct_16_xx": ("fg_pct_16ft_to_3pt", "float"),
    "pct_ast_fg2": ("assisted_share_2pt", "float"),
    "pct_ast_fg3": ("assisted_share_3pt", "float"),
    "pct_fga_dunk": ("dunk_share", "float"),
    "pct_fg3a_corner3": ("corner_three_share", "float"),
    "fg_pct_corner3": ("corner_three_pct", "float"),
}
PLAY_BY_PLAY_COLUMNS = {
    "pct_1": ("minutes_share_pg", "pct_points"),
    "pct_2": ("minutes_share_sg", "pct_points"),
    "pct_3": ("minutes_share_sf", "pct_points"),
    "pct_4": ("minutes_share_pf", "pct_points"),
    "pct_5": ("minutes_share_c", "pct_points"),
    "drawn_shooting": ("shooting_fouls_drawn_per_game", "season_total"),
    "and1s": ("and_ones_per_game", "season_total"),
    "drawn_offensive": ("charges_drawn_per_game", "season_total"),
    "tov_bad_pass": ("bad_pass_turnovers_per_game", "season_total"),
    "tov_lost_ball": ("lost_ball_turnovers_per_game", "season_total"),
    "astd_pts": ("points_created_by_assists_per_game", "season_total"),
}


def season_end_year(season: str) -> int:
    """Return 2026 for '2025-26' (Basketball-Reference names seasons by the year they end)."""
    start = int(season.split("-")[0])
    return start + 1


def table_url(season: str, kind: str) -> str:
    return f"https://www.basketball-reference.com/leagues/NBA_{season_end_year(season)}_{kind}.html"


def fetch(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text, encoding="utf-8")
    time.sleep(3)  # stay well under the site's rate limit
    return response.text


def find_table(html: str, table_id: str):
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=table_id)
    if table is not None:
        return table
    # Some Sports-Reference tables are shipped inside HTML comments.
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        if table_id in comment:
            table = BeautifulSoup(comment, "lxml").find("table", id=table_id)
            if table is not None:
                return table
    raise ValueError(f"table {table_id!r} not found")


def parse_value(raw: str, kind: str):
    raw = raw.strip()
    if raw == "":
        return None
    if kind == "int":
        return int(raw)
    if raw.endswith("%"):
        raw = raw[:-1]
    value = float(raw)
    return value / 100.0 if kind == "pct_points" else value


def parse_rows(table, columns: dict[str, tuple[str, str]]) -> list[dict]:
    """Return one dict per source row, in source order. Rows without a player id (e.g. League Average) are skipped."""
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        name_cell = tr.find("td", attrs={"data-stat": "name_display"})
        if name_cell is None or not name_cell.get("data-append-csv"):
            continue
        cells = {td["data-stat"]: td.get_text() for td in tr.find_all("td")}
        row = {
            "player_id": name_cell["data-append-csv"],
            "player_name": name_cell.get_text().strip(),
            "team": cells.get("team_name_abbr", "").strip() or None,
            "position": cells.get("pos", "").strip() or None,
        }
        for stat, (column, kind) in columns.items():
            row[column] = parse_value(cells.get(stat, ""), kind)
        games = parse_value(cells.get("games", ""), "int")
        for stat, (column, kind) in columns.items():
            if kind == "season_total" and row[column] is not None:
                row[column] = round(row[column] / games, 3) if games else None
        rows.append(row)
    return rows


def collapse_traded(rows: list[dict]) -> dict[str, dict]:
    """Keep one row per player.

    Basketball-Reference lists a traded player's combined row ('2TM', '3TM') first, followed by one
    row per team. We keep the combined row as the season total and record the individual teams.
    """
    by_player: dict[str, dict] = {}
    for row in rows:
        pid = row["player_id"]
        if pid not in by_player:
            by_player[pid] = dict(row, teams=[] if _is_combined(row["team"]) else [row["team"]])
        elif _is_combined(by_player[pid]["team"]) and row["team"]:
            by_player[pid]["teams"].append(row["team"])
    return by_player


def _is_combined(team: str | None) -> bool:
    return bool(team) and team.endswith("TM") and team[0].isdigit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", default=config.DEFAULT_SEASON)
    args = parser.parse_args()

    per_game_url = table_url(args.season, "per_game")
    advanced_url = table_url(args.season, "advanced")
    per_game_html = fetch(per_game_url, config.RAW_DIR / f"bbref_{args.season}_per_game.html")
    advanced_html = fetch(advanced_url, config.RAW_DIR / f"bbref_{args.season}_advanced.html")
    shooting_url = table_url(args.season, "shooting")
    play_by_play_url = table_url(args.season, "play-by-play")
    shooting_html = fetch(shooting_url, config.RAW_DIR / f"bbref_{args.season}_shooting.html")
    play_by_play_html = fetch(play_by_play_url, config.RAW_DIR / f"bbref_{args.season}_play-by-play.html")

    per_game = collapse_traded(parse_rows(find_table(per_game_html, "per_game_stats"), PER_GAME_COLUMNS))
    advanced = collapse_traded(parse_rows(find_table(advanced_html, "advanced"), ADVANCED_COLUMNS))
    shooting = collapse_traded(parse_rows(find_table(shooting_html, "shooting"), SHOOTING_COLUMNS))
    play_by_play = collapse_traded(parse_rows(find_table(play_by_play_html, "pbp_stats"), PLAY_BY_PLAY_COLUMNS))

    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect(config.DB_PATH)
    init_schema(conn)

    extra_tables = {
        "advanced": (advanced, ADVANCED_COLUMNS),
        "shooting": (shooting, SHOOTING_COLUMNS),
        "play-by-play": (play_by_play, PLAY_BY_PLAY_COLUMNS),
    }
    missing = {name: 0 for name in extra_tables}
    with conn:
        conn.execute("DELETE FROM player_season_stats WHERE season = ?", (args.season,))
        for pid, row in per_game.items():
            extra_values = {}
            for name, (table_rows, columns) in extra_tables.items():
                source_row = table_rows.get(pid)
                if source_row is None:
                    missing[name] += 1
                    source_row = {}
                extra_values.update({column: source_row.get(column) for column, _ in columns.values()})
            conn.execute(
                "INSERT INTO players (player_id, player_name, position) VALUES (?, ?, ?) "
                "ON CONFLICT(player_id) DO UPDATE SET player_name = excluded.player_name, position = excluded.position",
                (pid, row["player_name"], row["position"]),
            )
            record = {
                "player_id": pid,
                "season": args.season,
                "team": row["team"],
                "teams": ",".join(row["teams"]) or None,
                "position": row["position"],
                **{column: row[column] for column, _ in PER_GAME_COLUMNS.values()},
                **extra_values,
                "source_name": SOURCE_NAME,
                "source_url": " ; ".join([per_game_url, advanced_url, shooting_url, play_by_play_url]),
                "retrieved_at": retrieved_at,
            }
            columns = ", ".join(record)
            placeholders = ", ".join("?" for _ in record)
            conn.execute(f"INSERT INTO player_season_stats ({columns}) VALUES ({placeholders})", tuple(record.values()))

    print(f"Loaded {len(per_game)} players for {args.season} into {config.DB_PATH}")
    for name, count in missing.items():
        if count:
            print(f"Warning: {count} players had no {name} row; those columns are NULL.")


if __name__ == "__main__":
    main()
