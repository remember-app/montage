from src import config
from src.integrations.ai.contract import AnswerError, Blob, Provider, Reply, Usage
from src.integrations.ai.gemini import GeminiProvider
from src.integrations.ai.openrouter import OpenRouterProvider

__all__ = ["AnswerError", "Blob", "Provider", "Reply", "Usage", "ask", "provider"]


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
) -> Reply:
    return provider().ask(
        instruction, prompt, blobs, previous_interaction_id=previous_interaction_id
    )
