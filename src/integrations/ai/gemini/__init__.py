import http.client
import json
import logging
import urllib.error
import urllib.request

from src import config
from src.integrations.ai.contract import AnswerError, Blob, Provider, Reply

log = logging.getLogger("integrations.ai.gemini")


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


class GeminiProvider(Provider):
    def ask(
        self,
        instruction: str,
        prompt: str,
        blobs: tuple[Blob, ...] = (),
        previous_interaction_id: str | None = None,
    ) -> Reply:
        """One turn of the stored interaction; the instruction goes only with the first one."""
        parts = []
        if previous_interaction_id is None:
            parts.append({"type": "text", "text": instruction})
        parts.append({"type": "text", "text": prompt})
        parts += [
            {"type": blob.kind, "mime_type": blob.mime_type, "data": blob.encoded()}
            for blob in blobs
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
        return Reply(answer, usage, str(identifier) if identifier else None)
