"""Project configuration. Values can be overridden with environment variables (see .env.example)."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DB_PATH = Path(os.getenv("FULLCOURT_DB_PATH", PROCESSED_DIR / "nba.sqlite"))
INDEX_PATH = Path(os.getenv("FULLCOURT_INDEX_PATH", PROCESSED_DIR / "descriptions.faiss"))
INDEX_META_PATH = INDEX_PATH.with_suffix(".meta.json")

# The single season loaded for the MVP. Seasons use the "2025-26" label throughout.
DEFAULT_SEASON = os.getenv("FULLCOURT_DEFAULT_SEASON", "2025-26")

EMBEDDING_MODEL = os.getenv("FULLCOURT_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# bge models expect this instruction on queries (not on passages).
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# LLM provider: "ollama" (local, default), "anthropic", or "none" (rule-based parser + template
# explanations). If the provider is unreachable the app falls back to "none" and says so.
LLM_PROVIDER = os.getenv("FULLCOURT_LLM_PROVIDER", "ollama")
_DEFAULT_MODELS = {"ollama": "qwen2.5:7b", "anthropic": "claude-opus-5-5", "none": ""}
LLM_MODEL = os.getenv("FULLCOURT_LLM_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, ""))
OLLAMA_HOST = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.getenv("FULLCOURT_OLLAMA_NUM_CTX", "8192"))

# Ranking defaults. These are starting points to be evaluated, not tuned optima.
STAT_WEIGHT = float(os.getenv("FULLCOURT_STAT_WEIGHT", "0.6"))
SEMANTIC_WEIGHT = float(os.getenv("FULLCOURT_SEMANTIC_WEIGHT", "0.4"))
# "zero": a player with no style evidence gets no semantic credit.
# "neutral": a player with no style evidence gets the pool's median semantic score.
MISSING_SEMANTIC_POLICY = os.getenv("FULLCOURT_MISSING_SEMANTIC_POLICY", "zero")

# Default eligibility so that tiny samples (e.g. 2 games) don't dominate rate stats.
# Shown in the UI and the search plan as a default, never presented as a user constraint.
DEFAULT_MIN_GAMES = int(os.getenv("FULLCOURT_DEFAULT_MIN_GAMES", "20"))
DEFAULT_MIN_MINUTES = float(os.getenv("FULLCOURT_DEFAULT_MIN_MINUTES", "10"))

RESULT_LIMIT = 10
