import time

from docrag.embeddings.base import Embedder
from docrag.models import Hit
from docrag.retrieval.base import Retrieval, Retriever
from docrag.vectorstore.base import VectorStore


class DenseRetriever(Retriever):
    """Semantic search: embed the question and find the nearest chunks."""

    def __init__(self, store: VectorStore, embedder: Embedder):
        self._store, self._embedder = store, embedder

    def retrieve(self, question: str, k: int, docs: set[str] | None = None) -> Retrieval:
        t0 = time.perf_counter()
        vector = self._embedder.embed_query(question)
        t1 = time.perf_counter()
        found = self._store.search(vector, k, docs)
        hits = [
            Hit(chunk=h.chunk, score=h.score, dense_rank=rank, dense_score=h.score)
            for rank, h in enumerate(found, 1)
        ]
        return Retrieval(hits, embed_s=t1 - t0, search_s=time.perf_counter() - t1)

    def docs(self) -> set[str]:
        return self._store.docs()
