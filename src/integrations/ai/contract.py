import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Blob:
    mime_type: str
    data: bytes

    @property
    def kind(self) -> str:
        return self.mime_type.split("/")[0]

    def encoded(self) -> str:
        return base64.b64encode(self.data).decode()


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class Reply:
    text: str
    usage: Usage | None = None
    interaction_id: str | None = None


class AnswerError(RuntimeError):
    def __init__(self, message: str, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class Provider(ABC):
    """One provider of model replies: Gemini now, OpenRouter when it lands."""

    name: str = ""

    @property
    def model(self) -> str:
        return ""

    @abstractmethod
    def ask(
        self,
        instruction: str,
        prompt: str,
        blobs: tuple[Blob, ...] = (),
        previous_interaction_id: str | None = None,
    ) -> Reply:
        """Answer one turn; the reply carries the id that continues the conversation."""
