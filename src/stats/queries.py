"""Parameterized statistics queries.

Column names come only from schemas.METRICS; every user-supplied value is a bound parameter.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from src.schemas import METRICS, HardFilter, Position

STAT_COLUMNS = [m.column for m in METRICS.values()]


@dataclass(frozen=True)
class Eligibility:
    """Default sample-size floor. Distinct from user hard filters and reported separately in the UI."""

    min_games: int = 0
    min_minutes: float = 0.0


def available_seasons(conn: sqlite3.Connection) -> list[str]:
    return [row[0] for row in conn.execute("SELECT DISTINCT season FROM player_season_stats ORDER BY season DESC")]


def fetch_candidates(
    conn: sqlite3.Connection,
    season: str,
    positions: list[Position] | None = None,
    hard_filters: list[HardFilter] | None = None,
    eligibility: Eligibility = Eligibility(),
) -> list[dict]:
    """Return every player-season row that satisfies the season, positions, eligibility and hard filters.

    A NULL value never satisfies a hard filter on that metric (missing data is not assumed to pass).
    """
    clauses = ["s.season = ?"]
    params: list[object] = [season]

    if positions:
        clauses.append(f"s.position IN ({', '.join('?' for _ in positions)})")
        params.extend(p.value for p in positions)
    if eligibility.min_games:
        clauses.append("s.games_played >= ?")
        params.append(eligibility.min_games)
    if eligibility.min_minutes:
        clauses.append("s.minutes_per_game >= ?")
        params.append(eligibility.min_minutes)
    for f in hard_filters or []:
        column = METRICS[f.metric].column  # KeyError for anything not approved
        clauses.append(f"s.{column} IS NOT NULL AND s.{column} {f.op} ?")
        params.append(f.value)

    sql = (
        "SELECT s.player_id, p.player_name, s.season, s.team, s.teams, s.position, s.source_name, s.source_url, "
        + ", ".join(f"s.{c}" for c in STAT_COLUMNS)
        + " FROM player_season_stats s JOIN players p USING (player_id) WHERE "
        + " AND ".join(clauses)
    )
    return [dict(row) for row in conn.execute(sql, params)]


def find_player(conn: sqlite3.Connection, name: str, season: str) -> dict | None:
    """Look up a player in a season by case-insensitive name, used for 'similar to X' queries."""
    from unicodedata import normalize

    def fold(text: str) -> str:
        return normalize("NFKD", text).encode("ascii", "ignore").decode().lower().replace(".", "").strip()

    target = fold(name)
    rows = conn.execute(
        "SELECT p.player_id, p.player_name FROM players p JOIN player_season_stats s USING (player_id) WHERE s.season = ?",
        (season,),
    ).fetchall()
    exact = [r for r in rows if fold(r["player_name"]) == target]
    if exact:
        return dict(exact[0])
    partial = [r for r in rows if target and target in fold(r["player_name"])]
    return dict(partial[0]) if len(partial) == 1 else None
