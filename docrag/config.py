from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_GEMINI_MODELS = (
    "gemini-3.1-flash-lite,gemini-3.5-flash-lite,gemini-3-flash-preview,"
    "gemini-3.5-flash,gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash,gemini-2.5-pro"
)


class Settings(BaseSettings):
    """All configuration, read from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    google_api_key: str | None = None

    # Comma-separated fallback chain, tried in order (next model on 429/404/503).
    gemini_models: str = DEFAULT_GEMINI_MODELS
    # Open-weights model served through the Gemini API, to show what a
    # local-class model can do.
    # gemma-3-12b-it, the first choice, is no longer offered by the API (404). Of the Gemma models
    # this key can see, the 26B one (its name says 4B active parameters, a mixture of experts) is the closest
    # to a local-class model.
    gemma_model: str = "gemma-4-26b-a4b-it"
    openai_chat_model: str = "gpt-4o-mini"
    gemini_timeout_s: float = 120.0  # per request; the API accepts 10 s or more

    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "gemini-embedding-001"

    pdf_dir: Path = Path("pdfs")
    index_dir: Path = Path("data/index")
    embedding_cache_path: Path = Path("data/embedding_cache.sqlite")
    eval_dir: Path = Path("data/eval")

    # Where vectors live: "numpy" (files in index_dir, the default) or "pgvector" (Postgres).
    store: str = "numpy"
    database_url: str | None = None  # only needed for the pgvector store

    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_chars: int = 120  # shorter chunks are merged into the next one on the page
    top_k: int = 4
    retriever: str = "dense"  # dense | keyword | hybrid
    hybrid_pool: int = 20  # candidates each side contributes before fusion
    rerank_pool: int = 8  # candidates handed to the reranker, which keeps the top_k best
    # Best hit below this cosine score => "insufficient" without calling the LLM.
    # 0 disables it. Calibrate on the golden set once real embeddings exist.
    min_score: float = 0.0
    max_attempts: int = 3  # LLM tries to produce schema-valid JSON

    @property
    def gemini_model_chain(self) -> list[str]:
        return [m.strip() for m in self.gemini_models.split(",") if m.strip()]
