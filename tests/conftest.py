import hashlib

import numpy as np

from docrag.embeddings.base import Embedder


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder. Counts texts it was asked to embed."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake:hash"

    def embed_documents(self, texts):
        self.calls += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        return out
