from __future__ import annotations

from .config import Settings


def build_lm(settings: Settings, upstream_model: str, temperature: float, max_tokens: int):
    import dspy

    if not settings.upstream_api_key:
        raise ValueError("Missing upstream API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")

    return dspy.LM(
        model=f"openai/{upstream_model}",
        api_key=settings.upstream_api_key,
        api_base=settings.upstream_api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )
