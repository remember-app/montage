"""The cloud model behind the montage: one interface, two providers, and a journal of every call.

Three things the worker asks a general-purpose cloud model (doc-03): what is in a photo or a clip
and who is in it, what is said in a video, and in what order the pieces make a film. Which company
answers is a setting — `AI_PROVIDER=gemini|openai` — and nothing above this module knows which one
it was: the callers (understanding, speech, the director) see `AI` and its three methods.

Every one of those calls goes through `AI`, and every call through `AI` is written to `ai_requests`
(`schema.sql`) before its answer reaches the caller — that is the whole point of the gateway, and
why no caller is given a provider to talk to directly. Read it with psql:

    select created_at, provider, operation, ref, status, request, response from ai_requests
    order by id desc limit 20;

No key in the environment — no call. The gateway logs that it skipped one and answers None, which a
caller falls back from (the montage is then cut by the worker's own choosing, as it is today); what
it never does is go to the network or let anyone believe a model spoke.

Another provider: a class with `name`, `generate` and `transcribe`, and one line in `PROVIDERS`.
Its key and model then come from `<NAME>_API_KEY` and `<NAME>_MODEL` on their own.
"""

import base64
import json
import logging
import os
import time
import urllib.request
import uuid
from dataclasses import dataclass

from psycopg.types.json import Jsonb

TIMEOUT_SECONDS = 180  # a model watching a clip takes its time, and the job it serves is long

log = logging.getLogger("montage.ai")


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


# --- the providers ----------------------------------------------------------


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
        return json.loads(self.transport(request))

    def generate(self, instruction, prompt, blobs=(), as_json=True):
        """The model's answer as text, with `blobs` attached to the prompt."""
        raise NotImplementedError

    def transcribe(self, audio):
        """The speech in one audio file as plain text; empty when there is none."""
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
            # Asking for JSON in the prompt alone gets it wrapped in a ```json fence as often as not.
            body["response_format"] = {"type": "text", "mime_type": "application/json"}
        request = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
        )
        return _gemini_text(self._answered(request))

    def transcribe(self, audio):
        return self.generate(TRANSCRIBE, "", [audio], as_json=False).strip()


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
        return _openai_text(self._answered(request))

    def transcribe(self, audio):
        content_type, body = _multipart(
            {"model": self.transcribe_model, "response_format": "json"},
            "file", f"speech{audio.extension}", audio,
        )
        request = urllib.request.Request(
            self.TRANSCRIPTIONS,
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": content_type},
        )
        return self._answered(request).get("text", "").strip()


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
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
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
            name, ", ".join(PROVIDERS),
        )
        return None
    key = os.environ.get(f"{name.upper()}_API_KEY", "").strip()
    if not key:
        log.warning(
            "%s is chosen but %s_API_KEY is not in the environment: every call will be skipped",
            name, name.upper(),
        )
        return None
    return provider(key, os.environ.get(f"{name.upper()}_MODEL"), transport)


# --- what the worker asks ---------------------------------------------------
# The wording lives here and not in the callers: a prompt is what makes the answer comparable
# between the two providers, and swapping the provider must not change what was asked.

DESCRIBE = (
    "Ты разбираешь материал, снятый людьми на одном мероприятии. Опиши одним-двумя предложениями, "
    "что происходит в кадре, и перечисли, кто в нём есть, метками действующих лиц. Метку бери из "
    "списка уже известных, если это тот же человек или то же животное; если в кадре кто-то, кого в "
    "списке нет, заведи следующую по счёту: «человек-N» для человека, «животное-N» для животного. "
    'Ответ — JSON вида {"description": "...", "cast": ["человек-1", "животное-1"]}.'
)

TRANSCRIBE = (
    "Расшифруй речь в этой записи: верни только то, что сказано, слово в слово, без комментариев и "
    "без пометок о том, кто говорит. Речи не слышно — верни пустой ответ."
)

PLAN = (
    "Ты режиссёр монтажа. Из описанных кусков материала собери план ролика: какие куски войдут, в "
    "каком порядке, сколько секунд каждый и чем он важен. Держись истории, если она задана, и "
    "укладывайся в запрошенную длину. Кусок, помеченный must_include, обязан войти в план. "
    "Под ролик играет музыка из music — треки подряд, в этом порядке и такой длины; в music_note "
    "пожелания организатора, как их использовать, — учитывай их в порядке и длине кусков. Список "
    "music пустой — музыки в ролике нет вовсе, звучит живой звук самих видео. "
    'Ответ — JSON вида {"plan": [{"media_id": "...", "seconds": 3.5, "accent": "..."}]}.'
)


class AI:
    """The one door to the cloud model: three questions, a journal row for each, no provider in
    sight above it.

    A caller holds the worker's own connection and this gateway, asks one of the three, and gets
    either an answer or None — None meaning nothing was asked of anyone, because there is nobody
    to ask. Anything that goes wrong on the way to a configured model raises, so the job retries
    on it like on any other failure.
    """

    def __init__(self, conn, provider=None):
        """`provider` is for the tests; the worker takes what the environment says."""
        self.conn = conn
        self.provider = provider or from_env()

    # --- the three questions ---

    def describe_media(self, media, cast=(), ref=None):
        """What is in these frames, and who — one photo, or the frames of one clip.

        `cast` is everyone already named in this memory. It goes into the prompt so that the model
        recognises the ones it has met and introduces only the ones it has not: that is how the
        same person keeps one label across a whole memory without a face database of our own
        (doc-03, decision-9).

        Returns {"description": ..., "cast": [...]} or None when no model is configured.
        """
        known = ", ".join(cast) if cast else "пока никого"
        prompt = f"Уже известные действующие лица этого воспоминания: {known}."
        return self._call(
            "describe_media", DESCRIBE, prompt, media, ref,
            lambda provider: _as_json(provider.generate(DESCRIBE, prompt, media)),
        )

    def transcribe_speech(self, audio, ref=None):
        """What is said in one clip, word for word, as plain text.

        Who says it is deliberately not asked and deliberately not kept (decision-11). Silence
        comes back as an empty string — not as None, which here means nobody was asked at all.
        """
        return self._call(
            "transcribe_speech", TRANSCRIBE, "", [audio], ref,
            lambda provider: provider.transcribe(audio),
        )

    def plan_montage(self, shots, story="", target_seconds=None, music=(), music_note="",
                     ref=None):
        """The order of the film: which piece, for how long, and what it is there for.

        `shots` is what may go into the montage, one dict per piece — `media_id` and whatever is
        known about it (its description, its speech, `must_include`). A piece the organizer struck
        out is simply not in this list: the surest way for it not to be in the plan is for the
        model never to see it.

        `music` is the montage's soundtrack as it will actually sound — the tracks in the order
        they play and how long each of them runs, empty when the order asked for no music at all —
        and `music_note` is the organizer's own word on what to do with them. The soundtrack is
        assembled from the order's list whatever comes back: these two are the model's to plan the
        picture against, not to change (doc-03).

        Returns {"plan": [{"media_id": ..., "seconds": ..., "accent": ...}]} or None.
        """
        prompt = json.dumps(
            {"story": story, "target_seconds": target_seconds, "music": list(music),
             "music_note": music_note, "shots": list(shots)},
            ensure_ascii=False,
        )
        return self._call(
            "plan_montage", PLAN, prompt, (), ref,
            lambda provider: _as_json(provider.generate(PLAN, prompt)),
        )

    # --- the gateway itself ---

    def _call(self, operation, instruction, prompt, blobs, ref, ask):
        """Nothing reaches a model except through here, and nothing gets back to the caller before
        the journal holds both halves of it.

        The row is opened before the call and closed after it, rather than written once at the end:
        a worker killed mid-call, or a provider that never answers, still leaves the record of what
        was asked — which is what the journal is for (doc-03, «все запросы обязательно
        сохраняются»).
        """
        # What the journal keeps of the question. No key in it: the journal outlives every key
        # that was ever used and is read by hand in psql. No media bytes either — the material is
        # in the bucket already, and a copy of it in jsonb would make this the biggest table here.
        request = {
            "instruction": instruction,
            "prompt": prompt,
            "attached": [blob.summary() for blob in blobs],
        }

        if self.provider is None:
            log.warning(
                "%s skipped: no cloud model is configured, nothing was asked and nothing answered",
                operation,
            )
            # A call nobody made belongs in the journal too, so that «what did we ask the model»
            # has one answer and not two. It stays without a `response` for exactly that reason.
            self._born(
                operation, request, ref, status="skipped",
                provider=os.environ.get("AI_PROVIDER", "").strip().lower() or "unset",
                model=None, error="no cloud model is configured",
            )
            return None

        row = self._born(
            operation, request, ref, status="sent",
            provider=self.provider.name, model=self.provider.model,
        )
        started = time.monotonic()
        try:
            answer = ask(self.provider)
        except Exception as exc:
            self._closed(row, "failed", started, error=str(exc))
            raise
        self._closed(row, "answered", started, response=answer)
        return answer

    def _born(self, operation, request, ref, *, provider, model, status, error=None):
        return self.conn.execute(
            """INSERT INTO ai_requests (provider, model, operation, ref, status, request, error)
               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (provider, model, operation, ref, status, Jsonb(request), error),
        ).fetchone()[0]

    def _closed(self, row, status, started, response=None, error=None):
        self.conn.execute(
            """UPDATE ai_requests SET status = %s, response = %s, error = %s, took_ms = %s,
                                      updated_at = now()
               WHERE id = %s""",
            (
                status,
                Jsonb(response) if response is not None else None,
                error,
                round((time.monotonic() - started) * 1000),
                row,
            ),
        )


def _as_json(answer):
    """The model's words as the object they were asked to be. A model that answered in prose has
    failed the call, and the text of it goes into the journal as the error — that is the only
    place anyone will look for why.

    An object and not merely valid JSON: every caller here reads fields off what comes back, and a
    bare `null` or a list parses perfectly well and then fails three handlers down as an
    `AttributeError` about `NoneType`, with the answer that caused it nowhere to be seen.

    `parse_constant` because `json.loads` accepts `NaN` and `Infinity` and `jsonb` accepts neither:
    one of those in an answer would be paid for, then break the journal's own UPDATE two lines
    later — and the retry would pay for it again. They are malformed here, like prose, and the row
    keeps the text that caused it.
    """
    try:
        parsed = json.loads(answer, parse_constant=_no_such_number)
    except (ValueError, TypeError):  # a JSONDecodeError is a ValueError, and so is the one below
        parsed = None
    if not isinstance(parsed, dict):
        raise RuntimeError(f"the model did not answer with a JSON object: {str(answer)[:500]}")
    return parsed


def _no_such_number(constant):
    raise ValueError(constant)
