from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMResponse(BaseModel):
    text: str
    model: str  # the model that actually answered (matters with fallback chains)
    input_tokens: int | None = None  # as reported by the provider; None if it did not say
    output_tokens: int | None = None  # includes reasoning tokens where the provider bills them


class LLMCall(BaseModel):
    """One LLM call as seen by the caller, for tracing."""

    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float  # wall time, including any skipped models in a fallback chain
    purpose: str = "answer"  # "answer", or "rerank" for the reranker's calls


class AllModelsFailed(RuntimeError):
    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        detail = "; ".join(f"{m}: {e}" for m, e in errors.items())
        super().__init__(f"No model in the chain could answer. {detail}")


class LLM(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, system: str | None = None, json_mode: bool = False
    ) -> LLMResponse:
        """json_mode asks the provider to emit JSON where it supports that. It is a hint:
        callers must still validate, since some models (e.g. Gemma) cannot enforce it."""
