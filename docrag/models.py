from pathlib import Path

from pydantic import BaseModel


class Page(BaseModel):
    doc: str
    page: int  # 1-based
    text: str
    # (offset in text, heading path from the PDF outline), sorted; each runs to the next
    spans: list[tuple[int, str]] = []


class Chunk(BaseModel):
    id: str
    doc: str
    page: int
    text: str  # verbatim page text; citations are checked against this
    section: str | None = None

    def embedding_text(self) -> str:
        """What gets embedded: the text with its document and section as context."""
        header = " | ".join(part for part in (Path(self.doc).stem, self.section) if part)
        return f"{header}\n{self.text}"


class Hit(BaseModel):
    chunk: Chunk
    score: float  # the retriever's own ranking score, higher is better (cosine, BM25 or fused)
    # Where the chunk stood in each underlying ranking (1-based), None if it was not in that list.
    dense_rank: int | None = None
    dense_score: float | None = None  # cosine similarity
    keyword_rank: int | None = None
    keyword_score: float | None = None  # BM25
    rerank_score: int | None = None  # 0-3 from the LLM reranker, if one ran
