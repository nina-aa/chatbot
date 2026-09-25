from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Turns text into vectors. Implementations must be interchangeable."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable id such as 'openai:text-embedding-3-small'. Keys caches and indexes."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Returns float32 array of shape (len(texts), dim)."""

    def embed_query(self, text: str) -> np.ndarray:
        """Returns float32 array of shape (dim,)."""
        return self.embed_documents([text])[0]


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
