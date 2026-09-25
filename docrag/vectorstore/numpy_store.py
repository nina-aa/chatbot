"""Exact cosine-similarity search in memory. Plenty for a few thousand chunks."""

import json
from pathlib import Path

import numpy as np

from docrag.models import Chunk, Hit
from docrag.vectorstore.base import VectorStore


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class NumpyStore(VectorStore):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        self._chunks.extend(chunks)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, query_vector: np.ndarray, k: int, docs: set[str] | None = None) -> list[Hit]:
        if self._matrix is None:
            return []
        scores = self._matrix @ _normalize(np.asarray(query_vector, dtype=np.float32))
        if docs is not None:
            scores = np.where([c.doc in docs for c in self._chunks], scores, -np.inf)
        top = [i for i in np.argsort(-scores)[:k] if np.isfinite(scores[i])]
        return [Hit(chunk=self._chunks[i], score=float(scores[i])) for i in top]

    def docs(self) -> set[str]:
        return {c.doc for c in self._chunks}

    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._matrix)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(chunk.model_dump_json() + "\n")

    @classmethod
    def load(cls, directory: Path) -> "NumpyStore":
        store = cls()
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.model_validate_json(line) for line in f]
        store.add(chunks, np.load(directory / "vectors.npy"))
        return store

    def __len__(self) -> int:
        return len(self._chunks)
