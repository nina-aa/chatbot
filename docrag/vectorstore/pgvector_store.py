"""Exact cosine-similarity search in Postgres with the pgvector extension.

Same results as NumpyStore: vectors are stored L2-normalized, the score is
1 - cosine distance (= cosine similarity, higher is better), and ties are broken by
insertion order. The scan is exact (no HNSW/IVFFlat index), which keeps results
identical; an approximate index would be faster at scale but could change the top-k.

Several indexes (different embedders or chunk settings) share one table, told apart
by `index_name`.
"""

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from docrag.models import Chunk, Hit
from docrag.vectorstore.base import VectorStore
from docrag.vectorstore.numpy_store import _normalize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id         BIGSERIAL PRIMARY KEY,   -- insertion order
    index_name TEXT NOT NULL,
    chunk_id   TEXT NOT NULL,
    doc        TEXT NOT NULL,
    page       INTEGER NOT NULL,
    section    TEXT,
    text       TEXT NOT NULL,
    embedding  vector NOT NULL,         -- no fixed dimension: indexes of different embedders coexist
    UNIQUE (index_name, chunk_id)
);
CREATE INDEX IF NOT EXISTS chunks_index_doc ON chunks (index_name, doc);
"""


class PgVectorStore(VectorStore):
    def __init__(self, database_url: str, index_name: str) -> None:
        self._index = index_name
        self._conn = psycopg.connect(database_url, autocommit=True)
        # The extension must exist before the vector type can be registered.
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.execute(_SCHEMA)
        register_vector(self._conn)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        rows = [
            (self._index, c.id, c.doc, c.page, c.section, c.text, v)
            for c, v in zip(chunks, vectors)
        ]
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (index_name, chunk_id, doc, page, section, text, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                rows,
            )

    def search(self, query_vector: np.ndarray, k: int, docs: set[str] | None = None) -> list[Hit]:
        if k <= 0:
            return []
        query = _normalize(np.asarray(query_vector, dtype=np.float32))
        where, params = "index_name = %s", [self._index]
        if docs is not None:
            where += " AND doc = ANY(%s)"
            params.append(sorted(docs))
        # The distance is computed once, in the subquery. Putting the vector parameter in the
        # select list as well as in ORDER BY made each search about 10x slower (about 50 ms
        # instead of about 5 ms on 562 chunks).
        rows = self._conn.execute(
            f"SELECT chunk_id, doc, page, section, text, 1 - dist FROM ("
            f"SELECT id, chunk_id, doc, page, section, text, embedding <=> %s AS dist "
            f"FROM chunks WHERE {where}) t ORDER BY dist, id LIMIT %s",
            [query, *params, k],
        ).fetchall()
        return [
            Hit(chunk=Chunk(id=cid, doc=doc, page=page, section=section, text=text), score=float(score))
            for cid, doc, page, section, text, score in rows
        ]

    def docs(self) -> set[str]:
        rows = self._conn.execute("SELECT DISTINCT doc FROM chunks WHERE index_name = %s", [self._index])
        return {doc for (doc,) in rows}

    def chunks(self) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT chunk_id, doc, page, section, text FROM chunks WHERE index_name = %s ORDER BY id",
            [self._index],
        )
        return [Chunk(id=cid, doc=doc, page=page, section=section, text=text) for cid, doc, page, section, text in rows]

    def reset(self) -> None:
        """Remove this index's rows, so a rebuild replaces it like saving over a directory does."""
        self._conn.execute("DELETE FROM chunks WHERE index_name = %s", [self._index])

    def close(self) -> None:
        self._conn.close()

    # The interface was written for file-based stores. Here rows are written by add() and
    # the store is opened from a connection, so there is nothing to save or load by path.
    def save(self, directory) -> None:
        """No-op: add() already wrote to the database."""

    @classmethod
    def load(cls, directory) -> "PgVectorStore":
        raise NotImplementedError("Open it with PgVectorStore(database_url, index_name)")

    def __len__(self) -> int:
        return self._conn.execute("SELECT count(*) FROM chunks WHERE index_name = %s", [self._index]).fetchone()[0]
