from src.integrations.ai.contract import Blob, Provider, Reply, Tool, ToolResult


class OpenRouterProvider(Provider):
    """Placeholder: the contract is in place, the transport is not written yet."""

    name = "openrouter"

    def ask(
        self,
        instruction: str,
        prompt: str,
        blobs: tuple[Blob, ...] = (),
        previous_interaction_id: str | None = None,
        tools: tuple[Tool, ...] = (),
        tool_results: tuple[ToolResult, ...] = (),
        background: bool = False,
        stream: bool = False,
    ) -> Reply:
        raise NotImplementedError("the openrouter provider is not implemented yet")
