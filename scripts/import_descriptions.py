"""Collect source-backed playing-style descriptions from Wikipedia and load them into SQLite.

For every player who meets the default eligibility in the selected season, this script:
  1. resolves the player's English Wikipedia article via the search API (exact-name match only),
  2. looks for a "Player profile" / "Playing style" / "Style of play" section,
  3. stores that section's prose as paragraph-sized chunks with the article URL, revision id and
     revision timestamp.

Players whose article cannot be matched unambiguously, or whose article has no such section, get
no description. The app reports that style evidence is unavailable for them; nothing is generated.

Wikipedia text is licensed CC BY-SA 4.0. Each chunk keeps the revision-pinned URL for attribution.
Responses are cached in data/raw/wikipedia/ (gitignored).

Usage:
    python scripts/import_descriptions.py --season 2025-26
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.stats.database import connect, init_schema  # noqa: E402

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "FullCourt-research-assistant/0.1 (personal portfolio project; https://github.com/)"
CACHE_DIR = config.RAW_DIR / "wikipedia"
SECTION_PATTERN = re.compile(r"^(player profile|playing style|style of play|scouting report)$", re.I)
TARGET_WORDS = 160
MAX_WORDS = 260

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def api_get(params: dict, cache_key: str) -> dict:
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    params = {**params, "format": "json", "formatversion": 2}
    for attempt in range(4):
        response = session.get(API, params=params, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** attempt * 2)
            continue
        response.raise_for_status()
        break
    else:
        response.raise_for_status()
    data = response.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(0.2)
    return data


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\(.*?\)", "", name).lower().replace(".", "")
    return " ".join(re.sub(r"[^a-z ]", " ", name).split())


def safe_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode())


def resolve_article(player_id: str, player_name: str) -> str | None:
    """Return the article title only when a search hit's title matches the player's name exactly."""
    data = api_get(
        {"action": "query", "list": "search", "srsearch": f"{player_name} NBA basketball player", "srlimit": 5},
        f"search_{player_id}",
    )
    target = normalize_name(player_name)
    for hit in data.get("query", {}).get("search", []):
        if normalize_name(hit["title"]) == target:
            return hit["title"]
    return None


def find_profile_section(player_id: str, title: str) -> tuple[str, str, int] | None:
    data = api_get({"action": "parse", "page": title, "prop": "sections|revid", "redirects": 1}, f"sections_{player_id}")
    parse = data.get("parse")
    if not parse:
        return None
    for section in parse["sections"]:
        if SECTION_PATTERN.match(section["line"].strip()):
            return section["index"], section["line"].strip(), parse["revid"]
    return None


def section_paragraphs(player_id: str, title: str, section_index: str) -> list[str]:
    data = api_get(
        {"action": "parse", "page": title, "prop": "text", "section": section_index, "redirects": 1, "disabletoc": 1},
        f"section_{player_id}_{section_index}",
    )
    soup = BeautifulSoup(data["parse"]["text"], "lxml")
    for junk in soup.select("sup, table, style, figure, .mw-editsection, .hatnote, .thumb, .reflist"):
        junk.decompose()
    paragraphs = []
    for p in soup.find_all("p"):
        text = re.sub(r"\s+", " ", p.get_text()).strip()
        text = re.sub(r"\[\d+\]|\[citation needed\]", "", text)
        if len(text.split()) >= 12:
            paragraphs.append(text)
    return paragraphs


def revision_timestamp(player_id: str, revid: int) -> str | None:
    data = api_get({"action": "query", "prop": "revisions", "revids": revid, "rvprop": "ids|timestamp"}, f"rev_{player_id}_{revid}")
    for page in data.get("query", {}).get("pages", []):
        for rev in page.get("revisions", []):
            return rev["timestamp"]
    return None


def chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    """Merge short paragraphs and split long ones so chunks are roughly TARGET_WORDS words."""
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph.split()) <= MAX_WORDS:
            pieces.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current: list[str] = []
        for sentence in sentences:
            current.append(sentence)
            if len(" ".join(current).split()) >= TARGET_WORDS:
                pieces.append(" ".join(current))
                current = []
        if current:
            pieces.append(" ".join(current))

    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1].split()) + len(piece.split()) <= TARGET_WORDS:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", default=config.DEFAULT_SEASON)
    parser.add_argument("--min-games", type=int, default=config.DEFAULT_MIN_GAMES)
    parser.add_argument("--min-minutes", type=float, default=config.DEFAULT_MIN_MINUTES)
    args = parser.parse_args()

    conn = connect(config.DB_PATH)
    init_schema(conn)
    players = conn.execute(
        "SELECT p.player_id, p.player_name FROM players p JOIN player_season_stats s USING (player_id) "
        "WHERE s.season = ? AND s.games_played >= ? AND s.minutes_per_game >= ? ORDER BY s.minutes_per_game DESC",
        (args.season, args.min_games, args.min_minutes),
    ).fetchall()

    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = {"unmatched": [], "no_profile_section": [], "with_description": []}
    rows = []
    for i, player in enumerate(players, 1):
        pid, name = player["player_id"], player["player_name"]
        title = resolve_article(pid, name)
        if title is None:
            report["unmatched"].append(name)
            continue
        found = find_profile_section(pid, title)
        if found is None:
            report["no_profile_section"].append(name)
            continue
        section_index, section_name, revid = found
        chunks = chunk_paragraphs(section_paragraphs(pid, title, section_index))
        if not chunks:
            report["no_profile_section"].append(name)
            continue
        published_at = revision_timestamp(pid, revid)
        url = f"https://en.wikipedia.org/w/index.php?title={title.replace(' ', '_')}&oldid={revid}#{section_name.replace(' ', '_')}"
        for n, chunk in enumerate(chunks):
            rows.append((
                f"{pid}-wiki-{n}", pid, chunk, f"Wikipedia: {title}", url, section_name, revid, published_at,
                None, "encyclopedia player profile", "CC BY-SA 4.0", retrieved_at,
            ))
        report["with_description"].append(name)
        if i % 25 == 0:
            print(f"  {i}/{len(players)} players checked, {len(report['with_description'])} with descriptions")

    with conn:
        conn.execute("DELETE FROM player_documents WHERE source_name LIKE 'Wikipedia:%'")
        conn.executemany("INSERT INTO player_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    report_path = config.PROCESSED_DIR / "description_coverage.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Checked {len(players)} players: {len(report['with_description'])} with descriptions ({len(rows)} chunks), "
        f"{len(report['no_profile_section'])} without a profile section, {len(report['unmatched'])} unmatched. "
        f"Coverage report: {report_path}"
    )


if __name__ == "__main__":
    main()
