from src.integrations.ai.contract import Blob, Provider, Reply


class OpenRouterProvider(Provider):
    """Placeholder: the contract is in place, the transport is not written yet."""

    def ask(
        self,
        instruction: str,
        prompt: str,
        blobs: tuple[Blob, ...] = (),
        previous_interaction_id: str | None = None,
    ) -> Reply:
        raise NotImplementedError("the openrouter provider is not implemented yet")
