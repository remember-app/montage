import json

from src import config
from src.integrations.ai.contract import AnswerError, Blob, Provider, Reply
from src.integrations.ai.gemini import GeminiProvider
from src.integrations.ai.openrouter import OpenRouterProvider

__all__ = ["AnswerError", "Blob", "Provider", "Reply", "as_json", "ask", "provider"]


def provider() -> Provider:
    name = config.ai_provider()
    if name == "gemini":
        return GeminiProvider()
    if name == "openrouter":
        return OpenRouterProvider()
    raise RuntimeError(f"unknown AI provider {name!r}")


def ask(
    instruction: str,
    prompt: str,
    blobs: tuple[Blob, ...] = (),
    previous_interaction_id: str | None = None,
) -> tuple[str, dict | None, str | None]:
    reply = provider().ask(
        instruction, prompt, blobs, previous_interaction_id=previous_interaction_id
    )
    return reply.text, reply.usage, reply.interaction_id


def as_json(
    instruction: str,
    prompt: str,
    blobs: tuple[Blob, ...] = (),
    previous_interaction_id: str | None = None,
) -> tuple[dict, dict | None, str | None]:
    text, usage, identifier = ask(
        instruction, prompt, blobs, previous_interaction_id=previous_interaction_id
    )
    try:
        parsed = json.loads(text)
    except ValueError:
        raise AnswerError(f"the model did not answer with JSON: {text[:500]}", usage) from None
    if not isinstance(parsed, dict):
        raise AnswerError(f"the model answered with {type(parsed).__name__}, not an object", usage)
    return parsed, usage, identifier
