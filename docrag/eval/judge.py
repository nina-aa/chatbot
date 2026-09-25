"""LLM-as-judge. Run by a different model family than the one that wrote the answer."""

from pydantic import BaseModel

from docrag.eval.golden import GoldenItem
from docrag.llm.base import LLM, LLMCall
from docrag.rag.pipeline import format_context
from docrag.rag.schema import RagResult
from docrag.rag.structured import generate_validated

JUDGE_SYSTEM = """You are a strict evaluator of answers about industrial equipment documentation.

You get a question, a reference answer written by an expert, the documentation excerpts the system was given, and the system's answer.

Decide two things:
- "correct": the system's answer states the same essential facts as the reference answer. Extra correct detail is fine. A missing key fact or any contradiction of the reference is not correct.
- "faithful": every factual claim in the system's answer is supported by the excerpts. One unsupported claim means not faithful, even if it happens to be true.

Reply with a single JSON object and nothing else:
{"correct": true, "faithful": true, "reason": "one short sentence"}"""


class JudgeVerdict(BaseModel):
    correct: bool
    faithful: bool
    reason: str


def judge_answer(
    judge: LLM,
    item: GoldenItem,
    result: RagResult,
    max_attempts: int = 3,
    calls: list[LLMCall] | None = None,
) -> tuple[JudgeVerdict, str]:
    """Returns (verdict, judge_model_that_answered). Only call for answered results.

    Judge calls are appended to `calls` for token/latency accounting."""
    prompt = (
        f"Question:\n{item.question}\n\n"
        f"Reference answer:\n{item.reference_answer}\n\n"
        f"Documentation excerpts given to the system:\n{format_context(result.hits)}\n\n"
        f"System's answer:\n{result.answer}"
    )
    verdict, model, _ = generate_validated(
        judge, JudgeVerdict, prompt, JUDGE_SYSTEM, max_attempts, calls
    )
    return verdict, model
