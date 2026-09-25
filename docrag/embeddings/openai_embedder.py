import numpy as np
from openai import OpenAI

from docrag.embeddings.base import Embedder, batched


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str | None, model: str, batch_size: int = 128):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for batch in batched(texts, self._batch_size):
            response = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), -1)
