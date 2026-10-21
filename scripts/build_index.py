"""Embed every row of player_documents and write the FAISS index plus its metadata file.

Usage:
    python scripts/build_index.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.retrieval.semantic import SemanticIndex, SentenceTransformerEmbedder  # noqa: E402
from src.stats.database import connect  # noqa: E402


def main() -> None:
    conn = connect(config.DB_PATH)
    documents = [
        dict(r)
        for r in conn.execute(
            "SELECT document_id, player_id, COALESCE(embedding_text, text) AS text, description_type "
            "FROM player_documents ORDER BY document_id"
        )
    ]
    if not documents:
        sys.exit("No documents found. Run scripts/import_descriptions.py first.")

    started = time.perf_counter()
    embedder = SentenceTransformerEmbedder(config.EMBEDDING_MODEL, config.EMBEDDING_QUERY_PREFIX)
    index = SemanticIndex.build(documents, embedder)
    index.save(config.INDEX_PATH, config.INDEX_META_PATH, config.EMBEDDING_MODEL)
    print(
        f"Indexed {len(index)} chunks from {len({d['player_id'] for d in documents})} players "
        f"with {config.EMBEDDING_MODEL} in {time.perf_counter() - started:.1f}s -> {config.INDEX_PATH}"
    )


if __name__ == "__main__":
    main()
