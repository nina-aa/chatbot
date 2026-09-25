import numpy as np
from google import genai
from google.genai import types

from docrag.embeddings.base import Embedder, batched


class GeminiEmbedder(Embedder):
    def __init__(self, api_key: str | None, model: str, batch_size: int = 100):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        vectors = []
        for batch in batched(texts, self._batch_size):
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            vectors.extend(e.values for e in response.embeddings)
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), -1)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]
