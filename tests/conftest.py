"""Test fixtures. All players and numbers here are synthetic test data, not NBA facts."""

from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.hybrid import ResearchEngine  # noqa: E402
from src.retrieval.semantic import SemanticIndex  # noqa: E402
from src.stats.database import connect, init_schema  # noqa: E402

SEASON = "2025-26"

# player_id, name, position, games, mpg, ppg, apg, ts, usage, ast_rate, tov_rate, 3P%, 3PA
PLAYERS = [
    ("guard1", "Fixture Guard One", "PG", 70, 34.0, 28.0, 8.0, 0.620, 0.320, 0.400, 0.120, 0.380, 8.0),
    ("guard2", "Fixture Guard Two", "SG", 65, 30.0, 18.0, 3.0, 0.580, 0.240, 0.150, 0.100, 0.400, 7.0),
    ("guard3", "Fixture Guard Three", "PG", 60, 28.0, 10.0, 9.0, 0.540, 0.180, 0.420, 0.150, 0.330, 3.0),
    ("wing1", "Fixture Wing", "SF", 72, 33.0, 22.0, 4.0, 0.600, 0.270, 0.180, 0.090, 0.370, 6.0),
    ("big1", "Fixture Big", "C", 68, 30.0, 16.0, 2.0, 0.650, 0.200, 0.100, 0.110, None, 0.2),
    ("bench1", "Fixture Bench", "SG", 5, 8.0, 3.0, 1.0, 0.700, 0.150, 0.100, 0.080, 0.500, 1.0),
]

DOCUMENTS = [
    ("guard1-doc-0", "guard1", "pick and roll ball handler who drives into the paint and finds open shooters"),
    ("guard3-doc-0", "guard3", "pass first point guard who drives and kicks to open shooters, rarely shoots"),
    ("wing1-doc-0", "wing1", "isolation scorer who creates his own shot with pull up jumpers off the dribble"),
    ("big1-doc-0", "big1", "rim protector and lob finisher who sets screens and rebounds"),
]


class HashingEmbedder:
    """Deterministic bag-of-words embedder for tests."""

    dim = 256

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                vectors[row, zlib.crc32(word.encode()) % self.dim] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors


@pytest.fixture
def conn():
    connection = connect(":memory:")
    init_schema(connection)
    for pid, name, pos, gp, mpg, ppg, apg, ts, usg, ast, tov, fg3, fg3a in PLAYERS:
        connection.execute("INSERT INTO players VALUES (?, ?, ?)", (pid, name, pos))
        connection.execute(
            "INSERT INTO player_season_stats (player_id, season, team, position, games_played, minutes_per_game, "
            "points_per_game, assists_per_game, true_shooting_pct, usage_rate, assist_rate, turnover_rate, "
            "three_point_pct, three_point_attempts_per_game, source_name, source_url, retrieved_at) "
            "VALUES (?, ?, 'TST', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'fixture', 'fixture://', 'now')",
            (pid, SEASON, pos, gp, mpg, ppg, apg, ts, usg, ast, tov, fg3, fg3a),
        )
    for doc_id, pid, text in DOCUMENTS:
        connection.execute(
            "INSERT INTO player_documents (document_id, player_id, text, source_name, source_url, source_section, "
            "description_type, license, retrieved_at) VALUES (?, ?, ?, 'Fixture source', 'fixture://doc', "
            "'Player profile', 'test fixture', 'n/a', 'now')",
            (doc_id, pid, text),
        )
    connection.commit()
    return connection


@pytest.fixture
def index():
    docs = [{"document_id": d, "player_id": p, "text": t} for d, p, t in DOCUMENTS]
    return SemanticIndex.build(docs, HashingEmbedder())


@pytest.fixture
def engine(conn, index):
    return ResearchEngine(conn, index)
