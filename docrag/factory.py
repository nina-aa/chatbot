"""Builds concrete components from a provider name. The only place that knows all providers."""

from docrag.config import Settings
from docrag.embeddings.base import Embedder
from docrag.embeddings.gemini_embedder import GeminiEmbedder
from docrag.embeddings.openai_embedder import OpenAIEmbedder
from docrag.llm.base import LLM
from docrag.llm.gemini_llm import GeminiLLM
from docrag.llm.openai_llm import OpenAILLM

EMBEDDER_PROVIDERS = ("openai", "gemini")
LLM_PROVIDERS = ("openai", "gemini", "gemma")


def make_embedder(provider: str, s: Settings) -> Embedder:
    if provider == "openai":
        return OpenAIEmbedder(s.openai_api_key, s.openai_embedding_model)
    if provider == "gemini":
        return GeminiEmbedder(s.google_api_key, s.gemini_embedding_model)
    raise ValueError(f"Unknown embedder '{provider}'. Choose from {EMBEDDER_PROVIDERS}")


def make_llm(provider: str, s: Settings) -> LLM:
    if provider == "openai":
        return OpenAILLM(s.openai_api_key, s.openai_chat_model)
    if provider == "gemini":
        return GeminiLLM(s.gemini_model_chain, s.google_api_key)
    if provider == "gemma":
        return GeminiLLM([s.gemma_model], s.google_api_key)
    raise ValueError(f"Unknown LLM '{provider}'. Choose from {LLM_PROVIDERS}")


def default_judge(llm_provider: str) -> str:
    """Judge from a different model family than the answerer, to avoid self-preference bias."""
    return "gemini" if llm_provider == "openai" else "openai"


def same_family(a: str, b: str) -> bool:
    family = lambda p: "openai" if p == "openai" else "google"  # gemini and gemma are both Google
    return family(a) == family(b)
