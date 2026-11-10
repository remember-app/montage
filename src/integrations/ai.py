import base64
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from src import config

log = logging.getLogger("integrations.ai")

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"


@dataclass(frozen=True)
class Blob:
    mime_type: str
    data: bytes

    @property
    def kind(self) -> str:
        return self.mime_type.split("/")[0]

    def encoded(self) -> str:
        return base64.b64encode(self.data).decode()


def _text(payload: dict) -> str:
    if payload.get("output_text"):
        return str(payload["output_text"])
    return "".join(
        part.get("text", "")
        for step in payload.get("steps", [])
        if step.get("type") == "model_output"
        for part in step.get("content", [])
        if part.get("type") == "text"
    )


class AnswerError(RuntimeError):
    def __init__(self, message: str, usage: dict | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def ask(instruction: str, prompt: str, blobs: tuple[Blob, ...] = ()) -> tuple[str, dict | None]:
    parts = [{"type": "text", "text": f"{instruction}\n\n{prompt}".strip()}]
    parts += [
        {"type": blob.kind, "mime_type": blob.mime_type, "data": blob.encoded()} for blob in blobs
    ]
    body = {
        "model": config.gemini_model(),
        "input": parts,
        "response_format": {"type": "text", "mime_type": "application/json"},
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"x-goog-api-key": config.gemini_api_key(), "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=config.ai_timeout_seconds()) as response:
            payload = json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"model request failed: {exc.code} {detail}") from None
    usage = payload.get("usage") if isinstance(payload, dict) else None
    answer = _text(payload if isinstance(payload, dict) else {})
    if not answer:
        raise AnswerError("the model answered with nothing", usage)
    return answer, usage


def as_json(
    instruction: str, prompt: str, blobs: tuple[Blob, ...] = ()
) -> tuple[dict, dict | None]:
    answer, usage = ask(instruction, prompt, blobs)
    try:
        parsed = json.loads(answer)
    except ValueError:
        raise AnswerError(f"the model did not answer with JSON: {answer[:500]}", usage) from None
    if not isinstance(parsed, dict):
        raise AnswerError(f"the model answered with {type(parsed).__name__}, not an object", usage)
    return parsed, usage
