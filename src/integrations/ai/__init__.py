from src import config
from src.integrations.ai.contract import (
    AnswerError,
    Blob,
    Provider,
    Reply,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
)
from src.integrations.ai.gemini import GeminiProvider
from src.integrations.ai.openrouter import OpenRouterProvider

__all__ = [
    "AnswerError",
    "Blob",
    "Provider",
    "Reply",
    "Tool",
    "ToolCall",
    "ToolResult",
    "Usage",
    "ask",
    "provider",
]


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
    tools: tuple[Tool, ...] = (),
    tool_results: tuple[ToolResult, ...] = (),
    background: bool = False,
    stream: bool = False,
) -> Reply:
    return provider().ask(
        instruction,
        prompt,
        blobs,
        previous_interaction_id=previous_interaction_id,
        tools=tools,
        tool_results=tool_results,
        background=background,
        stream=stream,
    )
