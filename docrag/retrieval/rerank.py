"""Reranking: an LLM re-orders a wider candidate pool so the best few reach the answer prompt."""

import logging

from pydantic import BaseModel, Field, model_validator

from docrag.llm.base import LLM, AllModelsFailed, LLMCall
from docrag.models import Hit
from docrag.retrieval.base import Retrieval, Retriever
from docrag.rag.structured import StructuredOutputError, generate_validated

log = logging.getLogger("docrag")

RERANK_SYSTEM = """You judge which documentation excerpts help answer a question about industrial equipment.

Rate every excerpt from 0 to 3:
3 = states the answer or the steps directly
2 = closely related and probably useful
1 = same general topic but does not help answer the question
0 = unrelated

Judge only from the excerpt text. A "Problem:" line that matches the question is not enough on its own: prefer the excerpt that also contains the cause and remedy.

Reply with a single JSON object and nothing else, rating every excerpt exactly once:
{"ratings": [{"id": 1, "score": 3}, {"id": 2, "score": 0}]}"""


class Rating(BaseModel):
    id: int
    score: int = Field(ge=0, le=3)


class Ratings(BaseModel):
    ratings: list[Rating]

    @model_validator(mode="after")
    def _no_duplicates(self):
        ids = [r.id for r in self.ratings]
        if len(ids) != len(set(ids)):
            raise ValueError("each excerpt must be rated exactly once")
        return self


def format_excerpts(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        where = f"{c.doc}, page {c.page}" + (f", section {c.section}" if c.section else "")
        blocks.append(f"[{i}] ({where})\n{c.text}")
    return "\n\n".join(blocks)


class LLMReranker:
    def __init__(self, llm: LLM, max_attempts: int = 2):
        self._llm, self._max_attempts = llm, max_attempts

    def rerank(self, question: str, hits: list[Hit], calls: list[LLMCall]) -> list[Hit]:
        """Hits ordered by the LLM's score, best first. Ties keep the original order.

        Excerpts the LLM forgot to rate score 0, and ids outside the list are ignored.
        """
        prompt = f"Question: {question}\n\nExcerpts:\n{format_excerpts(hits)}"
        parsed, _, _ = generate_validated(
            self._llm, Ratings, prompt, RERANK_SYSTEM, self._max_attempts, calls, purpose="rerank"
        )
        scores = {r.id: r.score for r in parsed.ratings}
        scored = [
            h.model_copy(update={"rerank_score": scores.get(i, 0)}) for i, h in enumerate(hits, 1)
        ]
        return sorted(scored, key=lambda h: -h.rerank_score)  # sorted() is stable


class RerankingRetriever(Retriever):
    """Fetches `pool` candidates from another retriever, reranks them, returns the top k.

    If the reranker fails (every model down, or unusable output) the original order is
    kept, so a reranking problem degrades quality instead of failing the question.
    """

    def __init__(self, base: Retriever, reranker: LLMReranker, pool: int = 8):
        self._base, self._reranker, self._pool = base, reranker, pool

    def retrieve(self, question: str, k: int, docs: set[str] | None = None) -> Retrieval:
        found = self._base.retrieve(question, max(self._pool, k), docs)
        if len(found.hits) <= 1:
            return found
        calls: list[LLMCall] = []
        try:
            ordered = self._reranker.rerank(question, found.hits, calls)
        except (AllModelsFailed, StructuredOutputError) as e:
            log.warning("reranking failed, keeping the original order: %s", e)
            ordered = found.hits
        return Retrieval(
            ordered[:k], embed_s=found.embed_s, search_s=found.search_s,
            llm_calls=found.llm_calls + calls,
        )

    def docs(self) -> set[str]:
        return self._base.docs()
