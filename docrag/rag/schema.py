"""The contract between the LLM and the rest of the system."""

from typing import Literal

from pydantic import BaseModel, model_validator

from docrag.models import Hit
from docrag.rag.trace import Trace


class Statement(BaseModel):
    """One claim in the answer with the passage the LLM says supports it.

    Not trusted until the quote is verified against the retrieved context.
    """

    text: str
    doc: str
    page: int
    quote: str


class LLMAnswer(BaseModel):
    sufficient: bool
    statements: list[Statement] = []

    @model_validator(mode="after")
    def _answered_means_cited(self):
        if self.sufficient and not self.statements:
            raise ValueError("sufficient is true but statements is empty")
        if self.sufficient and any(not s.text.strip() for s in self.statements):
            raise ValueError("a statement has empty text")
        return self


class Citation(BaseModel):
    doc: str
    page: int
    section: str | None
    quote: str


class AnswerStatement(BaseModel):
    """A statement whose quote was found on the cited page of a retrieved chunk."""

    text: str
    citation: Citation


InsufficientReason = Literal[
    "low_retrieval_score",  # nothing relevant enough was retrieved; LLM not called
    "model_reported_insufficient",  # the LLM said the context doesn't answer it
    "no_verifiable_citations",  # the LLM claimed an answer but no quote checks out
]


class RagResult(BaseModel):
    status: Literal["answered", "insufficient", "error", "needs_device"]
    answer: str | None = None  # verified statements joined; only set when status == "answered"
    statements: list[AnswerStatement] = []  # verified only, each with its own citation
    citations: list[Citation] = []  # the distinct citations of those statements
    rejected_statements: list[Statement] = []  # claimed by the LLM, quote not found in context
    reason: InsufficientReason | str | None = None  # for "insufficient" / "error"
    model: str | None = None  # which model answered (None if the LLM was not called)
    attempts: int = 0  # LLM calls used, >1 means output had to be repaired
    hits: list[Hit] = []
    trace: Trace = Trace()
    devices: list[str] = []  # for "needs_device": the devices the user can choose from
