"""Disk cache so a chunk of text is only ever embedded once per embedding model."""

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from docrag.embeddings.base import Embedder


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedEmbedder(Embedder):
    """Wraps another embedder. Changing chunk size only re-embeds chunks whose text changed."""

    def __init__(self, inner: Embedder, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._inner = inner
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS vectors "
            "(model TEXT, key TEXT, dim INTEGER, vec BLOB, PRIMARY KEY (model, key))"
        )

    @property
    def name(self) -> str:
        return self._inner.name

    def _lookup(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):  # stay under SQLite's variable limit
            part = keys[i : i + 500]
            rows = self._db.execute(
                f"SELECT key, vec FROM vectors WHERE model = ? AND key IN ({','.join('?' * len(part))})",
                [self.name, *part],
            )
            found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return found

    def count_uncached(self, texts: list[str]) -> int:
        """How many unique texts would hit the API. Used for the cost estimate."""
        keys = {_key(t) for t in texts}
        return len(keys - self._lookup(list(keys)).keys())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        keys = [_key(t) for t in texts]
        cached = self._lookup(list(set(keys)))
        missing = {k: t for k, t in zip(keys, texts) if k not in cached}
        if missing:
            new = self._inner.embed_documents(list(missing.values()))
            with self._db:
                self._db.executemany(
                    "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?)",
                    [(self.name, k, v.shape[0], v.tobytes()) for k, v in zip(missing, new)],
                )
            cached.update(zip(missing, new))
        return np.stack([cached[k] for k in keys]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._inner.embed_query(text)  # queries are cheap and not cached
