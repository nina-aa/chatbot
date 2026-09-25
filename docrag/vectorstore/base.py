from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from docrag.models import Chunk, Hit


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int, docs: set[str] | None = None) -> list[Hit]:
        """Top-k chunks, best first. If docs is given, only chunks of those documents."""

    @abstractmethod
    def docs(self) -> set[str]:
        """Names of the documents in the store."""

    @abstractmethod
    def chunks(self) -> list[Chunk]:
        """Every chunk in the store, in insertion order."""

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "VectorStore": ...

    @abstractmethod
    def __len__(self) -> int: ...
