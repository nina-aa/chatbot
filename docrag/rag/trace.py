from pydantic import BaseModel, computed_field

from docrag.llm.base import LLMCall
from docrag.pricing import call_cost


def total_cost(calls: list[LLMCall]) -> float | None:
    """Sum of call costs, or None if any call has no known price (a partial sum would mislead)."""
    costs = [call_cost(c.model, c.input_tokens, c.output_tokens) for c in calls]
    return None if not calls or any(c is None for c in costs) else sum(costs)


class Trace(BaseModel):
    """Where the time and tokens of one question went."""

    embed_s: float = 0.0  # embedding the query
    search_s: float = 0.0  # vector search
    llm_calls: list[LLMCall] = []
    total_s: float = 0.0

    @computed_field
    @property
    def llm_s(self) -> float:
        return sum(c.latency_s for c in self.llm_calls)

    @computed_field
    @property
    def rerank_s(self) -> float:
        return sum(c.latency_s for c in self.llm_calls if c.purpose == "rerank")

    @computed_field
    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens or 0 for c in self.llm_calls)

    @computed_field
    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens or 0 for c in self.llm_calls)

    @computed_field
    @property
    def cost_usd(self) -> float | None:
        return total_cost(self.llm_calls)
