from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from docrag.llm.base import LLMCall
from docrag.models import Hit


@dataclass
class Retrieval:
    hits: list[Hit]
    embed_s: float = 0.0  # time spent embedding the query
    search_s: float = 0.0  # time spent searching
    llm_calls: list[LLMCall] = field(default_factory=list)  # calls a reranker made


class Retriever(ABC):
    """Finds the chunks most relevant to a question. Interchangeable behind the pipeline."""

    @abstractmethod
    def retrieve(self, question: str, k: int, docs: set[str] | None = None) -> Retrieval:
        """Top-k chunks, best first. If docs is given, only chunks of those documents."""

    @abstractmethod
    def docs(self) -> set[str]:
        """Names of the documents that can be searched."""
