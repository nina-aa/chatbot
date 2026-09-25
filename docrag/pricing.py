"""Price table for cost estimates. Verify against the provider's pricing page before quoting.

Models missing here (all Gemini/Gemma models: they run on a free tier for us) have no price,
so their cost is reported as unknown rather than guessed.
"""

# model -> (USD per 1M input tokens, USD per 1M output tokens)
CHAT_PRICE_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def call_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    price = CHAT_PRICE_PER_M.get(model)
    if price is None or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1e6
