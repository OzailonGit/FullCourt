"""Write a stat-based style profile for every player in a season into player_documents.

Run after scripts/import_stats.py and before scripts/build_index.py:
    python scripts/build_stat_profiles.py --season 2025-26

See src/stats/profiles.py for what the text contains and how the style tags are chosen.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.stats.database import connect, init_schema  # noqa: E402
from src.stats.profiles import DESCRIPTION_TYPE, TAG_COLUMNS, PoolPercentiles, build_profile_text, with_derived_columns  # noqa: E402

SOURCE_NAME = "Basketball-Reference stat profile"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", default=config.DEFAULT_SEASON)
    args = parser.parse_args()

    conn = connect(config.DB_PATH)
    init_schema(conn)
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT s.*, p.player_name FROM player_season_stats s JOIN players p USING (player_id) WHERE s.season = ?",
            (args.season,),
        )
    ]
    if not rows:
        sys.exit(f"No statistics for {args.season}. Run scripts/import_stats.py first.")

    def is_eligible(row: dict) -> bool:
        return (row["games_played"] or 0) >= config.DEFAULT_MIN_GAMES and (row["minutes_per_game"] or 0) >= config.DEFAULT_MIN_MINUTES

    pool = [with_derived_columns(r) for r in rows if is_eligible(r)]
    percentiles = PoolPercentiles(pool, TAG_COLUMNS)
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    documents = []
    for row in rows:
        documents.append((
            f"{row['player_id']}-stats-{args.season}",
            row["player_id"],
            build_profile_text(row, percentiles, is_eligible(row)),
            SOURCE_NAME,
            row["source_url"],
            f"Generated from {args.season} statistics",
            None,
            None,
            args.season,
            DESCRIPTION_TYPE,
            "Derived from Basketball-Reference data",
            retrieved_at,
        ))

    with conn:
        conn.execute(
            "DELETE FROM player_documents WHERE description_type = ? AND season = ?", (DESCRIPTION_TYPE, args.season)
        )
        conn.executemany("INSERT INTO player_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", documents)
    print(f"Wrote {len(documents)} stat-based profiles for {args.season} ({len(pool)} with style tags).")


if __name__ == "__main__":
    main()
