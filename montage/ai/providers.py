"""К кому воркер ходит за ответом: два провайдера за одним интерфейсом и один транспорт на обоих.

Какая компания отвечает — настройка `AI_PROVIDER=gemini|openai`, и выше этого модуля её имя нигде
не звучит. Ещё один провайдер — класс с `name`, `generate` и `transcribe` и строка в `PROVIDERS`;
ключ и модель к нему придут из `<NAME>_API_KEY` и `<NAME>_MODEL` сами.
"""

import base64
import json
import logging
import os
import urllib.request
import uuid
from dataclasses import dataclass, replace

from montage.ai.answers import counted, loaded
from montage.ai.prompts import TRANSCRIBE

TIMEOUT_SECONDS = 180  # a model watching a clip takes its time, and the job it serves is long

log = logging.getLogger("montage.ai.providers")


@dataclass(frozen=True)
class Blob:
    """A file on its way to a model: what it is, and its bytes.

    Bytes and not a path: both providers take the material inline, and the worker already has the
    original in a temp dir by the time it asks about it.
    """

    mime_type: str
    data: bytes

    @property
    def kind(self):
        """`image`, `audio` or `video` — the half of the mime type the providers switch on."""
        return self.mime_type.split("/")[0]

    @property
    def extension(self):
        """`.m4a` for `audio/m4a` — an upload whose decoder is picked by the file name needs one.
        From the subtype rather than `mimetypes`, which answers with whatever that machine's
        mime.types file happens to say and sometimes with nothing at all."""
        return "." + self.mime_type.split("/")[-1]

    def encoded(self):
        return base64.b64encode(self.data).decode()

    def summary(self):
        """What the journal keeps of it: enough to know what was sent, never the bytes."""
        return {"mime_type": self.mime_type, "bytes": len(self.data)}


@dataclass(frozen=True)
class Said:
    """A model's words, and what its provider counted the call at — the bill is paid in tokens.

    `usage` is the provider's own count, whole: a price can hang on more than the three numbers
    taken out of it (audio against text, a cached prefix, seconds instead of tokens). Each of the
    three is None when the provider did not say it — never a guess.
    """

    text: str
    usage: dict | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None  # everything billed at the output rate, thinking included
    total_tokens: int | None = None


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """A 30x is not an answer either of these APIs gives, and following one would take the key with
    it: urllib builds the next request out of *every* header but `content-*`, so `Authorization`
    and `x-goog-api-key` would be sent to whatever host the redirect names — a bucket somewhere, a
    proxy in the middle, anyone who can answer for those two endpoints once. Nothing here is worth
    that, so a redirect is refused: returning None leaves urllib to raise the 30x as the error it
    is, which fails the call and puts it in the journal like any other.
    """

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirects)


def _urlopen(request):
    with _opener.open(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


class Provider:
    """What the worker needs of a cloud model: an answer to a prompt, and a transcript of speech.

    `transport` is the only way bytes leave this process — a callable taking a prepared
    `urllib.request.Request` and returning the raw response body. It is a parameter and not a
    call to `urlopen` inside the adapters so that the tests can hand them answers recorded by
    hand: nothing in this file may reach the network on its own.
    """

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

    def generate(self, instruction, prompt, blobs=(), as_json=True):
        """The model's answer as `Said`, with `blobs` attached to the prompt."""
        raise NotImplementedError

    def transcribe(self, audio):
        """The speech in one audio file as `Said` plain text; empty when there is none."""
        raise NotImplementedError


class Gemini(Provider):
    """Google's Interactions API — one endpoint for text, images, audio and video alike.

    https://ai.google.dev/gemini-api/docs/interactions. The older `models/*:generateContent` is
    still served, but it is the legacy shape since June 2026 and new code is written against
    interactions. Video goes in whole, a second of it costing about as much as four photographs
    (doc-11) — which is why Gemini is the cheap provider of the two here.
    """

    name = "gemini"
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
    DEFAULT_MODEL = "gemini-3.8-flash"

    def generate(self, instruction, prompt, blobs=(), as_json=True):
        parts = [{"type": "text", "text": f"{instruction}\n\n{prompt}".strip()}]
        parts += [
            {"type": blob.kind, "mime_type": blob.mime_type, "data": blob.encoded()}
            for blob in blobs
        ]
        body = {"model": self.model, "input": parts}
        if as_json:
            # Asking for JSON in the prompt alone gets it wrapped in a ```json fence as often as
            # not.
            body["response_format"] = {"type": "text", "mime_type": "application/json"}
        request = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
        )
        return _gemini_said(self._answered(request))

    def transcribe(self, audio):
        said = self.generate(TRANSCRIBE, "", [audio], as_json=False)
        return replace(said, text=said.text.strip())


def _gemini_said(payload):
    usage = _usage(payload)
    output = counted(usage.get("total_output_tokens"))
    thought = counted(usage.get("total_thought_tokens")) or 0
    return Said(
        _gemini_text(payload),
        usage or None,
        counted(usage.get("total_input_tokens")),
        # Gemini counts thinking apart from the answer and bills both at the output rate
        # (interactions/thinking); OpenAI's output has its reasoning inside already.
        None if output is None else counted(output + thought),
        counted(usage.get("total_tokens")),
    )


def _gemini_text(payload):
    """What the model said in an interaction: the convenience field when it is there, and the text
    of its own step when it is not — an interaction also carries steps that are not the answer."""
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
    """OpenAI's Responses API, and its own endpoint for speech.

    https://developers.openai.com/api/reference. Two differences from Gemini that reach this file:
    speech is not a prompt but a file upload to `/v1/audio/transcriptions`, and video is not an
    input OpenAI takes at all — a clip is described by the frames pulled out of it, which is the
    caller's business and not ours (doc-11).
    """

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

    def generate(self, instruction, prompt, blobs=(), as_json=True):
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

    def transcribe(self, audio):
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
    """The same three counts in a response and in a transcription — the latter only when its model
    bills in tokens and not in seconds."""
    usage = _usage(payload)
    tokens = (counted(usage.get(key)) for key in ("input_tokens", "output_tokens", "total_tokens"))
    return Said(text, usage or None, *tokens)


def _usage(payload):
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else {}


def _openai_image(blob):
    if blob.kind != "image":
        # Loudly, and here rather than in a 400 from OpenAI: the caller has a clip in its hands and
        # has to know it is the one who must cut frames out of it.
        raise ValueError(
            f"openai takes no {blob.kind} in a prompt: cut a clip into frames, send speech to "
            "transcribe()"
        )
    # The data URI prefix is not decoration — without it the part is parsed as a URL and fails.
    return {"type": "input_image", "image_url": f"data:{blob.mime_type};base64,{blob.encoded()}"}


def _openai_text(payload):
    """The words out of a response. Its `output` is a list of items and the message is rarely the
    first of them — a reasoning item usually is."""
    return "".join(
        part.get("text", "")
        for item in payload.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def _multipart(fields, file_field, filename, blob):
    """One multipart/form-data body, by hand — a file upload is the shape the transcription
    endpoint takes, and nothing in the standard library builds one."""
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
    """The provider the setting names, with the key that goes with it — or None, out loud.

    None is not a failure: the worker runs without a cloud model, and it has to, because the key is
    the human's to give and the choice of provider his to make (decision-10). Every None here is a
    line in the log saying which of the two is missing.
    """
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
