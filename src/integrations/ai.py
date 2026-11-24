import base64
import http.client
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from src import config

log = logging.getLogger("integrations.ai")


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


def ask(
    instruction: str,
    prompt: str,
    blobs: tuple[Blob, ...] = (),
    previous_interaction_id: str | None = None,
) -> tuple[str, dict | None, str | None]:
    """One model turn; the returned interaction id continues the conversation on the next call.

    The instruction goes only with the first turn. A later turn carries the new prompt and blobs,
    while the server keeps the earlier inputs and outputs. Implicit caching works on its own, and a
    stateful continuation makes its hits more likely; explicit cache objects are not available on
    this endpoint.
    """
    parts = []
    if previous_interaction_id is None:
        parts.append({"type": "text", "text": instruction})
    parts.append({"type": "text", "text": prompt})
    parts += [
        {"type": blob.kind, "mime_type": blob.mime_type, "data": blob.encoded()} for blob in blobs
    ]
    body = {
        "model": config.gemini_model(),
        "input": parts,
        "response_format": {"type": "text", "mime_type": "application/json"},
    }
    if previous_interaction_id:
        body["previous_interaction_id"] = previous_interaction_id
    request = urllib.request.Request(
        config.ai_endpoint(),
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
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
        raise RuntimeError(f"model request failed: {exc}") from None
    usage = payload.get("usage") if isinstance(payload, dict) else None
    identifier = payload.get("id") if isinstance(payload, dict) else None
    answer = _text(payload if isinstance(payload, dict) else {})
    if not answer:
        raise AnswerError("the model answered with nothing", usage)
    return answer, usage, str(identifier) if identifier else None


def as_json(
    instruction: str,
    prompt: str,
    blobs: tuple[Blob, ...] = (),
    previous_interaction_id: str | None = None,
) -> tuple[dict, dict | None, str | None]:
    answer, usage, identifier = ask(
        instruction, prompt, blobs, previous_interaction_id=previous_interaction_id
    )
    try:
        parsed = json.loads(answer)
    except ValueError:
        raise AnswerError(f"the model did not answer with JSON: {answer[:500]}", usage) from None
    if not isinstance(parsed, dict):
        raise AnswerError(f"the model answered with {type(parsed).__name__}, not an object", usage)
    return parsed, usage, identifier


if __name__ == "__main__":
    import os
    from unittest import mock

    os.environ.setdefault("GEMINI_API_KEY", "test")
    seen = []

    class Reply:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake(request, timeout):
        seen.append(json.loads(request.data))
        return Reply(
            {"id": f"i{len(seen)}", "output_text": "{}", "usage": {"total_input_tokens": 1}}
        )

    with mock.patch("urllib.request.urlopen", fake):
        _, _, first_id = ask("be a director", "first")
        _, _, second_id = ask("be a director", "second", previous_interaction_id=first_id)
    assert (first_id, second_id) == ("i1", "i2")
    assert seen[0]["input"][0]["text"] == "be a director"
    assert "previous_interaction_id" not in seen[0]
    assert seen[1]["input"][0]["text"] == "second"
    assert seen[1]["previous_interaction_id"] == "i1"
    assert all(part.get("text") != "be a director" for part in seen[1]["input"])

    with mock.patch("urllib.request.urlopen", side_effect=http.client.RemoteDisconnected("closed")):
        try:
            ask("be a director", "third")
        except RuntimeError as exc:
            assert "model request failed" in str(exc)
        else:
            raise AssertionError("a dropped connection must fail loudly")
    print("ok")
