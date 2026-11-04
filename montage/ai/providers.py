import base64
import json
import logging
import os
import urllib.request
import uuid
from dataclasses import dataclass, replace

from montage.ai.answers import counted, loaded
from montage.ai.prompts import TRANSCRIBE

TIMEOUT_SECONDS = 180

log = logging.getLogger("montage.ai.providers")


@dataclass(frozen=True)
class Blob:
    mime_type: str
    data: bytes

    @property
    def kind(self):
        return self.mime_type.split("/")[0]

    @property
    def extension(self):
        return "." + self.mime_type.split("/")[-1]

    def encoded(self):
        return base64.b64encode(self.data).decode()

    def summary(self):
        return {"mime_type": self.mime_type, "bytes": len(self.data)}


@dataclass(frozen=True)
class Said:
    text: str
    usage: dict | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirects)


def _urlopen(request):
    with _opener.open(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


class Provider:
    name = ""
    DEFAULT_MODEL = ""

    def __init__(self, key, model=None, transport=_urlopen):
        self.key = key
        self.model = model or self.DEFAULT_MODEL
        self.transport = transport

    def __str__(self):
        return f"{self.name} ({self.model})"

    def _answered(self, request):
        return loaded(self.transport(request))

    def generate(self, instruction, prompt, blobs=(), as_json=True, least_thinking=False):
        raise NotImplementedError

    def transcribe(self, audio, least_thinking=False):
        raise NotImplementedError


class Gemini(Provider):
    name = "gemini"
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
    DEFAULT_MODEL = "gemini-3.8-flash"
    LEAST_THINKING = "low"

    def generate(self, instruction, prompt, blobs=(), as_json=True, least_thinking=False):
        parts = [{"type": "text", "text": f"{instruction}\n\n{prompt}".strip()}]
        parts += [
            {"type": blob.kind, "mime_type": blob.mime_type, "data": blob.encoded()}
            for blob in blobs
        ]
        body = {"model": self.model, "input": parts}
        if as_json:
            body["response_format"] = {"type": "text", "mime_type": "application/json"}
        if least_thinking:
            body["generation_config"] = {"thinking_level": self.LEAST_THINKING}
        request = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
        )
        return _gemini_said(self._answered(request))

    def transcribe(self, audio, least_thinking=False):
        said = self.generate(TRANSCRIBE, "", [audio], as_json=False, least_thinking=least_thinking)
        return replace(said, text=said.text.strip())


def _gemini_said(payload):
    usage = _usage(payload)
    output = counted(usage.get("total_output_tokens"))
    thought = counted(usage.get("total_thought_tokens")) or 0
    return Said(
        _gemini_text(payload),
        usage or None,
        counted(usage.get("total_input_tokens")),
        None if output is None else counted(output + thought),
        counted(usage.get("total_tokens")),
    )


def _gemini_text(payload):
    if payload.get("output_text"):
        return payload["output_text"]
    return "".join(
        part.get("text", "")
        for step in payload.get("steps", [])
        if step.get("type") == "model_output"
        for part in step.get("content", [])
        if part.get("type") == "text"
    )


class OpenAI(Provider):
    name = "openai"
    ENDPOINT = "https://api.openai.com/v1/responses"
    TRANSCRIPTIONS = "https://api.openai.com/v1/audio/transcriptions"
    DEFAULT_MODEL = "gpt-5.5"
    DEFAULT_TRANSCRIBE_MODEL = "gpt-4o-transcribe"

    def __init__(self, key, model=None, transport=_urlopen):
        super().__init__(key, model, transport)
        self.transcribe_model = (
            os.environ.get("OPENAI_TRANSCRIBE_MODEL") or self.DEFAULT_TRANSCRIBE_MODEL
        )

    def generate(self, instruction, prompt, blobs=(), as_json=True, least_thinking=False):
        content = [{"type": "input_text", "text": prompt}]
        content += [_openai_image(blob) for blob in blobs]
        body = {
            "model": self.model,
            "instructions": instruction,
            "input": [{"role": "user", "content": content}],
        }
        if as_json:
            body["text"] = {"format": {"type": "json_object"}}
        request = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        payload = self._answered(request)
        return _openai_said(_openai_text(payload), payload)

    def transcribe(self, audio, least_thinking=False):
        content_type, body = _multipart(
            {"model": self.transcribe_model, "response_format": "json"},
            "file",
            f"speech{audio.extension}",
            audio,
        )
        request = urllib.request.Request(
            self.TRANSCRIPTIONS,
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": content_type},
        )
        payload = self._answered(request)
        return _openai_said(payload.get("text", "").strip(), payload)


def _openai_said(text, payload):
    usage = _usage(payload)
    tokens = (counted(usage.get(key)) for key in ("input_tokens", "output_tokens", "total_tokens"))
    return Said(text, usage or None, *tokens)


def _usage(payload):
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else {}


def _openai_image(blob):
    if blob.kind != "image":
        raise ValueError(
            f"openai takes no {blob.kind} in a prompt: cut a clip into frames, send speech to "
            "transcribe()"
        )
    return {"type": "input_image", "image_url": f"data:{blob.mime_type};base64,{blob.encoded()}"}


def _openai_text(payload):
    return "".join(
        part.get("text", "")
        for item in payload.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def _multipart(fields, file_field, filename, blob):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode()
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {blob.mime_type}\r\n\r\n"
    ).encode()
    body += blob.data + f"\r\n--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", bytes(body)


PROVIDERS = {Gemini.name: Gemini, OpenAI.name: OpenAI}


def from_env(transport=_urlopen):
    name = os.environ.get("AI_PROVIDER", "").strip().lower()
    if not name:
        log.info("AI_PROVIDER is not set: no cloud model will be called")
        return None
    provider = PROVIDERS.get(name)
    if provider is None:
        log.error(
            "AI_PROVIDER=%s is not a provider this worker has (%s): no cloud model will be called",
            name,
            ", ".join(PROVIDERS),
        )
        return None
    key = os.environ.get(f"{name.upper()}_API_KEY", "").strip()
    if not key:
        log.warning(
            "%s is chosen but %s_API_KEY is not in the environment: every call will be skipped",
            name,
            name.upper(),
        )
        return None
    return provider(key, os.environ.get(f"{name.upper()}_MODEL"), transport)
