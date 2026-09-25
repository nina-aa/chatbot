"""Keyword search with Okapi BM25, written out so it can be read and tested."""

import math
import re
import time
from collections import Counter

from docrag.models import Chunk, Hit
from docrag.retrieval.base import Retrieval, Retriever

_TOKEN = re.compile(r"[a-z0-9]+")

# Filler words that say nothing about the topic. Without this, a chatty question such as
# "my probe dont work at all" matches half the manual on "my", "at" and "all".
# Fragments of contractions ("can't" -> "can", "t") are included. Numbers and codes are kept.
STOPWORDS = frozenset(
    """a about after all also am an and any are aren as at be because been before being both but by
    can cannot cant could couldn d did didn do does doesn doing don done down during each else for
    from further had hadn has hasn have haven having he her here hers him his how i if in into is
    isn it its itself just ll m me more most my myself no nor not now of off on once only or other
    our ours out over own re s same shall she should shouldn so some such t than that the their
    theirs them then there these they this those through to too under until up us ve very was
    wasn we were weren what when where which while who whom why will with won would wouldn you
    your yours""".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric runs. 'HTM10 03' -> ['htm10', '03'], '0706hex' stays whole."""
    return _TOKEN.findall(text.lower())


def content_terms(text: str) -> list[str]:
    """tokenize() without stopwords: the terms BM25 indexes and looks up."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def matched_terms(question: str, chunk: Chunk) -> list[str]:
    """The distinct query terms that occur in the chunk, in query order."""
    in_chunk = set(content_terms(chunk.embedding_text()))
    seen: list[str] = []
    for term in content_terms(question):
        if term in in_chunk and term not in seen:
            seen.append(term)
    return seen


class Bm25Index:
    """BM25 over a fixed set of chunks.

    Term statistics (document frequency, average length) come from exactly these chunks,
    so a device-restricted index weighs terms by how common they are in that manual.
    The indexed text is the chunk's embedding text: document name, section and body.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self._chunks, self._k1, self._b = chunks, k1, b
        self._length: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(chunk index, tf)]
        for i, chunk in enumerate(chunks):
            tf = Counter(content_terms(chunk.embedding_text()))
            self._length.append(sum(tf.values()))
            for term, n in tf.items():
                self._postings.setdefault(term, []).append((i, n))
        self._avg_length = (sum(self._length) / len(chunks)) if chunks else 0.0

    def _idf(self, term: str) -> float:
        n, df = len(self._chunks), len(self._postings[term])
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, question: str, k: int) -> list[tuple[int, float]]:
        """(chunk index, score) for the k best chunks with any matching term, best first."""
        scores: dict[int, float] = {}
        for term in set(content_terms(question)):
            if term not in self._postings:
                continue
            idf = self._idf(term)
            for i, tf in self._postings[term]:
                norm = self._k1 * (1 - self._b + self._b * self._length[i] / self._avg_length)
                scores[i] = scores.get(i, 0.0) + idf * tf * (self._k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))  # ties: document order
        return ranked[:k]

    def chunk(self, index: int) -> Chunk:
        return self._chunks[index]


class Bm25Retriever(Retriever):
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._indexes: dict[frozenset[str] | None, Bm25Index] = {}

    def _index_for(self, docs: set[str] | None) -> Bm25Index:
        key = None if docs is None else frozenset(docs)
        if key not in self._indexes:
            chosen = self._chunks if docs is None else [c for c in self._chunks if c.doc in docs]
            self._indexes[key] = Bm25Index(chosen)
        return self._indexes[key]

    def retrieve(self, question: str, k: int, docs: set[str] | None = None) -> Retrieval:
        t0 = time.perf_counter()
        index = self._index_for(docs)
        hits = [
            Hit(chunk=index.chunk(i), score=score, keyword_rank=rank, keyword_score=score)
            for rank, (i, score) in enumerate(index.search(question, k), 1)
        ]
        return Retrieval(hits, search_s=time.perf_counter() - t0)

    def docs(self) -> set[str]:
        return {c.doc for c in self._chunks}
