from google import genai
from google.genai import errors, types

from docrag.llm.base import LLM, AllModelsFailed, LLMResponse

# Quota exhausted (429), model not available to this key (404), model failing or overloaded
# (500, 502, 503) or not answering in time (504, which is also what our own timeout produces):
# all mean "this model cannot help right now", so we move on immediately.
SKIP_CODES = {404, 429, 500, 502, 503, 504}


class GeminiLLM(LLM):
    """Tries each model in order. A skippable error goes straight to the next model; a request
    that does not answer within `timeout_s` counts as one."""

    def __init__(
        self, models: list[str], api_key: str | None = None, client=None, timeout_s: float = 120.0
    ):
        if not models:
            raise ValueError("At least one model is required")
        if client is None:
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set")
            # Without a timeout one hung request blocks a whole run. The API refuses deadlines
            # under 10 s; an expired one comes back as a 504, which is skippable.
            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self._client = client
        self._models = models

    def generate(
        self, prompt: str, system: str | None = None, json_mode: bool = False
    ) -> LLMResponse:
        failures: dict[str, str] = {}
        for model in self._models:
            # Gemma rejects system instructions and JSON mode: fold the system prompt
            # into the user turn and rely on the caller's validation instead.
            if model.startswith("gemma"):
                contents = f"{system}\n\n{prompt}" if system else prompt
                config = types.GenerateContentConfig(temperature=0)
            else:
                contents = prompt
                config = types.GenerateContentConfig(
                    temperature=0,
                    system_instruction=system,
                    response_mime_type="application/json" if json_mode else None,
                )
            try:
                response = self._client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except errors.APIError as e:
                if e.code not in SKIP_CODES:
                    raise
                failures[model] = f"HTTP {e.code}"
                continue
            if not response.text:  # blocked or empty output
                failures[model] = "empty response"
                continue
            usage = getattr(response, "usage_metadata", None)
            out = None
            if usage is not None:  # reasoning ("thoughts") tokens are billed as output
                out = (getattr(usage, "candidates_token_count", None) or 0) + (
                    getattr(usage, "thoughts_token_count", None) or 0
                )
            return LLMResponse(
                text=response.text,
                model=model,
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=out,
            )
        raise AllModelsFailed(failures)
