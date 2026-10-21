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
    conn.commit()
