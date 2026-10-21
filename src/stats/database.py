"""SQLite schema and connection helpers.

Units: every percentage or rate is stored as a fraction in [0, 1] (e.g. 0.612 for 61.2% TS).
Per-game counting stats are stored as per-game averages. Missing values are NULL, never 0.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    player_id   TEXT PRIMARY KEY,      -- Basketball-Reference player slug, e.g. 'doncilu01'
    player_name TEXT NOT NULL,
    position    TEXT                   -- most recent listed position; season-specific value is in player_season_stats
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    player_id                     TEXT NOT NULL REFERENCES players(player_id),
    season                        TEXT NOT NULL,   -- '2025-26'
    team                          TEXT,            -- team abbreviation, or '2TM'/'3TM' for combined traded-player totals
    teams                         TEXT,            -- comma-separated teams played for, in source order
    position                      TEXT,            -- position listed by the source for this season
    age                           INTEGER,
    games_played                  INTEGER,
    games_started                 INTEGER,
    minutes_per_game              REAL,
    points_per_game               REAL,
    assists_per_game              REAL,
    rebounds_per_game             REAL,
    steals_per_game               REAL,
    blocks_per_game               REAL,
    turnovers_per_game            REAL,
    three_point_attempts_per_game REAL,
    field_goal_pct                REAL,
    three_point_pct               REAL,
    free_throw_pct                REAL,
    effective_field_goal_pct      REAL,
    true_shooting_pct             REAL,
    usage_rate                    REAL,
    assist_rate                   REAL,
    turnover_rate                 REAL,
    rebound_rate                  REAL,
    steal_rate                    REAL,
    block_rate                    REAL,
    box_plus_minus                REAL,            -- points per 100 possessions, not a fraction
    -- shooting table: where shots come from and how they are created
    average_shot_distance         REAL,            -- feet
    shot_share_0_3ft              REAL,            -- share of field-goal attempts
    shot_share_3_10ft             REAL,
    shot_share_10_16ft            REAL,
    shot_share_16ft_to_3pt        REAL,
    three_point_attempt_rate      REAL,            -- share of field-goal attempts that are threes
    fg_pct_0_3ft                  REAL,
    fg_pct_3_10ft                 REAL,
    fg_pct_10_16ft                REAL,
    fg_pct_16ft_to_3pt            REAL,
    assisted_share_2pt            REAL,            -- share of made twos that were assisted
    assisted_share_3pt            REAL,            -- share of made threes that were assisted
    dunk_share                    REAL,            -- share of field-goal attempts that are dunks
    corner_three_share            REAL,            -- share of three-point attempts from the corners
    corner_three_pct              REAL,
    -- play-by-play table: estimated position split and drawn/committed events
    minutes_share_pg              REAL,
    minutes_share_sg              REAL,
    minutes_share_sf              REAL,
    minutes_share_pf              REAL,
    minutes_share_c               REAL,
    shooting_fouls_drawn_per_game REAL,
    and_ones_per_game             REAL,
    charges_drawn_per_game        REAL,            -- offensive fouls drawn
    bad_pass_turnovers_per_game   REAL,
    lost_ball_turnovers_per_game  REAL,
    points_created_by_assists_per_game REAL,
    source_name                   TEXT NOT NULL,
    source_url                    TEXT NOT NULL,
    retrieved_at                  TEXT NOT NULL,
    PRIMARY KEY (player_id, season)
);

CREATE TABLE IF NOT EXISTS player_documents (
    document_id      TEXT PRIMARY KEY,
    player_id        TEXT NOT NULL REFERENCES players(player_id),
    text             TEXT NOT NULL,
    source_name      TEXT NOT NULL,
    source_url       TEXT NOT NULL,
    source_section   TEXT,
    revision_id      INTEGER,
    published_at     TEXT,            -- timestamp of the source revision, when known
    season           TEXT,            -- NULL when the text is not tied to a single season
    description_type TEXT NOT NULL,
    license          TEXT,
    retrieved_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_player ON player_documents(player_id);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    # check_same_thread=False: Streamlit reuses the cached connection across script-run threads.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
    conn.commit()


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Bring a database created by an older version of SCHEMA up to date.

    CREATE TABLE IF NOT EXISTS leaves existing tables untouched, so new columns are added here.
    """
    reference = sqlite3.connect(":memory:")
    reference.executescript(SCHEMA)
    tables = [row[0] for row in reference.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    for table in tables:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for _, name, column_type, *_ in reference.execute(f"PRAGMA table_info({table})"):
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {column_type}")
    reference.close()
