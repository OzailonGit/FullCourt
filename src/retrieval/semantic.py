"""Semantic retrieval over player description chunks (Sentence Transformers + FAISS).

Vectors are L2-normalized and stored in an IndexFlatIP, so inner product equals cosine similarity.
The metadata file maps each vector row back to its document_id and player_id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str], is_query: bool) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        if is_query:
            texts = [self.query_prefix + t for t in texts]
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return vectors.astype("float32")


@dataclass(frozen=True)
class ChunkHit:
    document_id: str
    player_id: str
    similarity: float
    description_type: str


@dataclass
class PlayerSemanticMatch:
    player_id: str
    similarity: float  # best chunk similarity for this player
    hits: list[ChunkHit]  # this player's chunks, most similar first
    best_by_type: dict[str, float] = field(default_factory=dict)


class SemanticIndex:
    def __init__(self, index: faiss.Index, metadata: list[dict], embedder: Embedder):
        if index.ntotal != len(metadata):
            raise ValueError("index and metadata are out of sync; rebuild with scripts/build_index.py")
        self.index = index
        self.metadata = metadata
        self.embedder = embedder

    @classmethod
    def build(cls, documents: list[dict], embedder: Embedder) -> SemanticIndex:
        """documents: dicts with document_id, player_id, text and (optionally) description_type."""
        vectors = embedder.encode([d["text"] for d in documents], is_query=False) if documents else np.zeros((0, 1), "float32")
        dim = vectors.shape[1] if len(documents) else 1
        index = faiss.IndexFlatIP(dim)
        if len(documents):
            index.add(vectors)
        metadata = [
            {"document_id": d["document_id"], "player_id": d["player_id"], "description_type": d.get("description_type", "description")}
            for d in documents
        ]
        return cls(index, metadata, embedder)

    @classmethod
    def load(cls, index_path: Path, meta_path: Path, embedder: Embedder) -> SemanticIndex:
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(index, meta["documents"], embedder)

    def save(self, index_path: Path, meta_path: Path, model_name: str) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        meta_path.write_text(json.dumps({"model": model_name, "documents": self.metadata}, indent=1), encoding="utf-8")

    def __len__(self) -> int:
        return self.index.ntotal

    def player_vector(self, player_id: str) -> np.ndarray | None:
        """Mean of a player's chunk vectors, re-normalized. Used for 'players similar to X'."""
        rows = [i for i, m in enumerate(self.metadata) if m["player_id"] == player_id]
        if not rows:
            return None
        vector = np.mean([self.index.reconstruct(i) for i in rows], axis=0)
        return (vector / np.linalg.norm(vector)).astype("float32")

    def search_text(self, query: str) -> dict[str, PlayerSemanticMatch]:
        return self.search_vector(self.embedder.encode([query], is_query=True)[0])

    def search_vector(self, vector: np.ndarray) -> dict[str, PlayerSemanticMatch]:
        """Score every chunk (the corpus is small) and group by player."""
        if len(self) == 0:
            return {}
        scores, rows = self.index.search(vector.reshape(1, -1).astype("float32"), len(self))
        matches: dict[str, PlayerSemanticMatch] = {}
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            meta = self.metadata[row]
            hit = ChunkHit(meta["document_id"], meta["player_id"], float(score), meta.get("description_type", "description"))
            match = matches.get(hit.player_id)
            if match is None:
                match = matches[hit.player_id] = PlayerSemanticMatch(hit.player_id, hit.similarity, [])
            match.hits.append(hit)  # results arrive in descending score order, so the first hit per type is its best
            match.best_by_type.setdefault(hit.description_type, hit.similarity)
        return matches
