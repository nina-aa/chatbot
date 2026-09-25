from openai import OpenAI

from docrag.llm.base import LLM, LLMResponse


class OpenAILLM(LLM):
    def __init__(self, api_key: str | None, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(
        self, prompt: str, system: str | None = None, json_mode: bool = False
    ) -> LLMResponse:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0, **extra
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=self._model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
