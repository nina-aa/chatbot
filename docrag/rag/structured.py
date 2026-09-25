"""Get schema-valid JSON out of an LLM, repairing invalid output by asking again."""

import json
import time

from pydantic import BaseModel, ValidationError

from docrag.llm.base import LLM, LLMCall


class StructuredOutputError(RuntimeError):
    def __init__(self, attempts: int, last_error: str):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"No valid output after {attempts} attempts: {last_error}")


def extract_json(text: str) -> object:
    """Find the first JSON value in text, tolerating code fences and surrounding prose."""
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        raise ValueError("no JSON found in reply")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def generate_validated[T: BaseModel](
    llm: LLM,
    schema: type[T],
    prompt: str,
    system: str | None,
    max_attempts: int = 3,
    calls: list[LLMCall] | None = None,
    purpose: str = "answer",
) -> tuple[T, str, int]:
    """Returns (parsed, model_that_answered, attempts_used).

    On invalid output the model is shown its own reply and the error, then retried.
    Every call made is appended to `calls`, so tokens spent on failed attempts are
    still counted even when this raises.
    """
    current = prompt
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        response = llm.generate(current, system=system, json_mode=True)
        if calls is not None:
            calls.append(
                LLMCall(
                    model=response.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_s=time.perf_counter() - started,
                    purpose=purpose,
                )
            )
        try:
            return schema.model_validate(extract_json(response.text)), response.model, attempt
        except (ValueError, ValidationError) as e:  # JSONDecodeError is a ValueError
            last_error = str(e)
            current = (
                f"{prompt}\n\nYour previous reply was invalid.\n"
                f"Previous reply:\n{response.text}\n\nProblem:\n{last_error}\n\n"
                "Reply again with only the corrected JSON object."
            )
    raise StructuredOutputError(max_attempts, last_error)
