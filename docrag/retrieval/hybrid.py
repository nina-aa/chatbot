"""Combining the dense and keyword rankings with reciprocal rank fusion."""

from docrag.models import Hit
from docrag.retrieval.base import Retrieval, Retriever

RRF_K = 60  # the constant from the original RRF paper; larger flattens the difference between ranks


def fuse(dense: list[Hit], keyword: list[Hit], k: int, rrf_k: int = RRF_K) -> list[Hit]:
    """Reciprocal rank fusion: a chunk scores the sum of 1 / (rrf_k + rank) over the lists it is in.

    Only ranks matter, so BM25 scores and cosine similarities need no common scale.
    A chunk found by both lists always beats a chunk found by one at a comparable rank.
    """
    merged: dict[str, Hit] = {}
    for h in dense:
        merged[h.chunk.id] = Hit(
            chunk=h.chunk, score=1.0 / (rrf_k + h.dense_rank),
            dense_rank=h.dense_rank, dense_score=h.dense_score,
        )
    for h in keyword:
        found = merged.get(h.chunk.id) or Hit(chunk=h.chunk, score=0.0)
        merged[h.chunk.id] = found.model_copy(
            update={
                "score": found.score + 1.0 / (rrf_k + h.keyword_rank),
                "keyword_rank": h.keyword_rank,
                "keyword_score": h.keyword_score,
            }
        )
    best_rank = lambda h: min(r for r in (h.dense_rank, h.keyword_rank) if r is not None)
    return sorted(merged.values(), key=lambda h: (-h.score, best_rank(h)))[:k]


class HybridRetriever(Retriever):
    """Dense plus keyword, fused. Each side contributes its top `pool` candidates."""

    def __init__(self, dense: Retriever, keyword: Retriever, pool: int = 20, rrf_k: int = RRF_K):
        self._dense, self._keyword, self._pool, self._rrf_k = dense, keyword, pool, rrf_k

    def retrieve(self, question: str, k: int, docs: set[str] | None = None) -> Retrieval:
        pool = max(self._pool, k)
        d = self._dense.retrieve(question, pool, docs)
        w = self._keyword.retrieve(question, pool, docs)
        return Retrieval(
            fuse(d.hits, w.hits, k, self._rrf_k), embed_s=d.embed_s, search_s=d.search_s + w.search_s
        )

    def docs(self) -> set[str]:
        return self._dense.docs()
