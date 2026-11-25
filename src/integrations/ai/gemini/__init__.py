import http.client
import json
import logging
import time
import urllib.error
import urllib.request

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

log = logging.getLogger("integrations.ai.gemini")

POLL_SECONDS = 2
TERMINAL = {
    "completed",
    "failed",
    "cancelled",
    "incomplete",
    "requires_action",
    "requires_continuation",
}


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


def _usage(payload: dict) -> Usage | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None

    def number(name: str) -> int | None:
        value = usage.get(name)
        return int(value) if isinstance(value, (int, float)) else None

    return Usage(
        input_tokens=number("total_input_tokens"),
        output_tokens=number("total_output_tokens"),
        total_tokens=number("total_tokens"),
        cached_input_tokens=number("total_cached_tokens"),
        reasoning_tokens=number("total_thought_tokens"),
    )


def _arguments(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _calls(payload: dict) -> tuple[ToolCall, ...]:
    calls = []
    for index, step in enumerate(payload.get("steps", [])):
        if not isinstance(step, dict) or step.get("type") != "function_call":
            continue
        calls.append(
            ToolCall(
                str(step.get("id") or f"call-{index}"),
                str(step.get("name") or ""),
                _arguments(step.get("arguments")),
            )
        )
    return tuple(calls)


def _result_parts(result: ToolResult) -> list[dict]:
    parts = []
    if result.text:
        parts.append({"type": "text", "text": result.text})
    parts += [
        {"type": blob.kind, "mime_type": blob.mime_type, "data": blob.encoded()}
        for blob in result.blobs
    ]
    return parts or [{"type": "text", "text": "ok"}]


def _failure(payload: dict) -> str:
    error = payload.get("error")
    if error is None:
        return "the interaction ended without a result"
    return json.dumps(error, ensure_ascii=False)[:500]


def _request(method: str, url: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"x-goog-api-key": config.gemini_api_key(), "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=config.ai_timeout_seconds()) as response:
            return json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"model request failed: {exc.code} {detail}") from None
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
        raise RuntimeError(f"model request failed: {exc}") from None


def _poll(identifier: str) -> dict:
    url = f"{config.ai_endpoint().rstrip('/')}/{identifier}"
    deadline = time.monotonic() + config.ai_background_seconds()
    while True:
        payload = _request("GET", url)
        status = str(payload.get("status") or "")
        if status in TERMINAL:
            return payload
        if time.monotonic() > deadline:
            raise RuntimeError(f"the background interaction {identifier} did not finish in time")
        time.sleep(POLL_SECONDS)


class GeminiProvider(Provider):
    name = "gemini"

    @property
    def model(self) -> str:
        return config.gemini_model()

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
        body = self._body(instruction, prompt, blobs, previous_interaction_id, tools, tool_results)
        if stream:
            payload = self._stream(body)
        elif background:
            payload = self._background(body)
        else:
            payload = _request("POST", config.ai_endpoint(), body)
        answer = _text(payload)
        usage = _usage(payload)
        identifier = payload.get("id")
        calls = _calls(payload)
        if not answer and not calls:
            raise AnswerError("the model answered with nothing", usage)
        return Reply(answer, usage, str(identifier) if identifier else None, calls)

    def _body(
        self,
        instruction: str,
        prompt: str,
        blobs: tuple[Blob, ...],
        previous_interaction_id: str | None,
        tools: tuple[Tool, ...],
        tool_results: tuple[ToolResult, ...],
    ) -> dict:
        parts = []
        if previous_interaction_id is None:
            parts.append({"type": "text", "text": instruction})
        if prompt:
            parts.append({"type": "text", "text": prompt})
        parts += [
            {"type": blob.kind, "mime_type": blob.mime_type, "data": blob.encoded()}
            for blob in blobs
        ]
        parts += [
            {
                "type": "function_result",
                "call_id": result.call_id,
                "name": result.name,
                "result": _result_parts(result),
                "is_error": result.is_error,
            }
            for result in tool_results
        ]
        body = {
            "model": self.model,
            "input": parts,
            "response_format": {"type": "text", "mime_type": "application/json"},
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in tools
            ]
        if previous_interaction_id:
            body["previous_interaction_id"] = previous_interaction_id
        return body

    def _stream(self, body: dict) -> dict:
        request = urllib.request.Request(
            config.ai_endpoint(),
            data=json.dumps({**body, "stream": True}).encode(),
            method="POST",
            headers={"x-goog-api-key": config.gemini_api_key(), "Content-Type": "application/json"},
        )
        texts: list[str] = []
        calls: dict[object, dict] = {}
        identifier = None
        usage = None
        status = None
        try:
            with urllib.request.urlopen(request, timeout=config.ai_timeout_seconds()) as response:
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except ValueError:
                        continue
                    kind = event.get("event_type")
                    if kind == "interaction.created":
                        interaction = event.get("interaction") or {}
                        identifier = interaction.get("id") or identifier
                    elif kind == "step.start":
                        step = event.get("step") or {}
                        if step.get("type") == "function_call":
                            index = event.get("index")
                            calls[index] = {
                                "id": str(step.get("id") or f"call-{index}"),
                                "name": str(step.get("name") or ""),
                                "arguments": step.get("arguments") or {},
                                "partial": "",
                            }
                    elif kind == "step.delta":
                        delta = event.get("delta") or {}
                        index = event.get("index")
                        if delta.get("type") == "text":
                            texts.append(str(delta.get("text") or ""))
                        elif delta.get("type") == "arguments_delta" and index in calls:
                            calls[index]["partial"] += str(
                                delta.get("arguments") or delta.get("text") or ""
                            )
                    elif kind == "interaction.completed":
                        interaction = event.get("interaction") or {}
                        identifier = interaction.get("id") or identifier
                        usage = interaction.get("usage")
                        status = interaction.get("status")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"model request failed: {exc.code} {detail}") from None
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
            raise RuntimeError(f"model request failed: {exc}") from None
        steps = []
        for entry in calls.values():
            arguments = entry["arguments"]
            if entry["partial"]:
                try:
                    arguments = json.loads(entry["partial"])
                except ValueError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            steps.append(
                {
                    "type": "function_call",
                    "id": entry["id"],
                    "name": entry["name"],
                    "arguments": arguments,
                }
            )
        return {
            "id": identifier,
            "status": status,
            "usage": usage,
            "output_text": "".join(texts),
            "steps": steps,
        }

    def _background(self, body: dict) -> dict:
        payload = _request("POST", config.ai_endpoint(), {**body, "background": True})
        identifier = payload.get("id")
        if not identifier:
            raise RuntimeError("the background interaction has no id")
        while True:
            payload = _poll(str(identifier))
            status = str(payload.get("status") or "")
            if status in ("completed", "requires_action"):
                return payload
            if status != "requires_continuation":
                raise RuntimeError(f"the interaction ended as {status!r}: {_failure(payload)}")
            token = payload.get("continuation_token")
            if not token:
                raise RuntimeError("the interaction requires continuation without a token")
            payload = _request(
                "POST",
                config.ai_endpoint(),
                {
                    "model": self.model,
                    "previous_interaction_id": identifier,
                    "continuation_token": token,
                    "background": True,
                },
            )
            identifier = payload.get("id") or identifier
