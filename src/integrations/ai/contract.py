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
class Reply:
    text: str
    usage: dict | None = None
    interaction_id: str | None = None


class AnswerError(RuntimeError):
    def __init__(self, message: str, usage: dict | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class Provider(ABC):
    """One provider of model replies: Gemini now, OpenRouter when it lands."""

    @abstractmethod
    def ask(
        self,
        instruction: str,
        prompt: str,
        blobs: tuple[Blob, ...] = (),
        previous_interaction_id: str | None = None,
    ) -> Reply:
        """Answer one turn; the reply carries the id that continues the conversation."""
