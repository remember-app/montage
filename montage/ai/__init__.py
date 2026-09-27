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

Which company answers is `providers.py`, and what comes back is read in `answers.py`; this file is
the door itself — what is asked, and the journal row that every question leaves behind.
"""

import json
import logging
import os
import time

from psycopg.types.json import Jsonb

from montage.ai import answers
from montage.ai.prompts import DESCRIBE, PLAN, TRANSCRIBE
from montage.ai.providers import Blob, Said, from_env

__all__ = ["AI", "DESCRIBE", "PLAN", "TRANSCRIBE", "Blob", "from_env"]

log = logging.getLogger("montage.ai")


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

        Returns the model's answer — asked for as {"description": ..., "cast": [...]}, and read as
        the untrusted thing it is (`understanding`) — or None when no model is configured.
        """
        known = ", ".join(cast) if cast else "пока никого"
        prompt = f"Уже известные действующие лица этого воспоминания: {known}."
        return self._call(
            "describe_media",
            DESCRIBE,
            prompt,
            media,
            ref,
            lambda provider: provider.generate(DESCRIBE, prompt, media),
            answers.as_json,
        )

    def transcribe_speech(self, audio, ref=None):
        """What is said in one clip, word for word, as plain text.

        Who says it is deliberately not asked and deliberately not kept (decision-11). Silence
        comes back as an empty string — not as None, which here means nobody was asked at all.
        """
        return self._call(
            "transcribe_speech",
            TRANSCRIBE,
            "",
            [audio],
            ref,
            lambda provider: provider.transcribe(audio),
        )

    def plan_montage(self, shots, story="", target_seconds=None, music=(), ref=None):
        """The order of the film: which piece, for how long, and what it is there for.

        `shots` is what may go into the montage, one dict per piece — `media_id` and whatever is
        known about it (its description, its speech, `must_include`). A piece the organizer struck
        out is simply not in this list: the surest way for it not to be in the plan is for the
        model never to see it.

        `music` is the montage's soundtrack as it will actually sound — the tracks in the order
        they play, how long each of them runs and the organizer's own wish on a track where they
        wrote one, empty when the order asked for no music at all. The soundtrack is assembled from
        the order's list whatever comes back: it is the model's to plan the picture against, not to
        change (doc-03).

        Returns the plan as it can be read (`answers.planned`) or None:

            {"overview": "...",
             "plan": [{"media_id", "start", "seconds", "live", "thought", "part",
                       "part_title", "focus"}],
             "left_out": {media_id: "почему не взял"}}

        The answer itself is kept whole in `ai_requests`; what comes back here is what survived
        being read — the model's own words about the film, and nothing it made up.
        """
        prompt = json.dumps(
            {
                "story": story,
                "target_seconds": target_seconds,
                "music": list(music),
                "shots": list(shots),
            },
            ensure_ascii=False,
        )
        return answers.planned(
            self._call(
                "plan_montage",
                PLAN,
                prompt,
                (),
                ref,
                lambda provider: provider.generate(PLAN, prompt),
                answers.as_json,
            )
        )

    # --- the gateway itself ---

    def _call(self, operation, instruction, prompt, blobs, ref, ask, read=None):
        """Nothing reaches a model except through here, and nothing gets back to the caller before
        the journal holds both halves of it.

        The row is opened before the call and closed after it, rather than written once at the end:
        a worker killed mid-call, or a provider that never answers, still leaves the record of what
        was asked — which is what the journal is for (doc-03, «все запросы обязательно
        сохраняются»).

        `ask` gets the provider's `Said`; `read` makes the answer of its words, which are the answer
        themselves without one. Apart, because words that fail to read were billed all the same:
        the failed row keeps what the provider counted them at.
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
                operation,
                request,
                ref,
                status="skipped",
                provider=os.environ.get("AI_PROVIDER", "").strip().lower() or "unset",
                model=None,
                error="no cloud model is configured",
            )
            return None

        row = self._born(
            operation,
            request,
            ref,
            status="sent",
            provider=self.provider.name,
            model=self.provider.model,
        )
        started = time.monotonic()
        said = None
        try:
            said = ask(self.provider)
            answer = read(said.text) if read else said.text
        except Exception as exc:
            self._closed(row, "failed", started, said, error=str(exc))
            raise
        self._closed(row, "answered", started, said, response=answer)
        return answer

    def _born(self, operation, request, ref, *, provider, model, status, error=None):
        return self.conn.execute(
            """INSERT INTO ai_requests (provider, model, operation, ref, status, request, error)
               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (provider, model, operation, ref, status, Jsonb(request), error),
        ).fetchone()[0]

    def _closed(self, row, status, started, said, response=None, error=None):
        """`said` is None when the provider never answered — and then nothing was counted."""
        said = said or Said("")
        self.conn.execute(
            """UPDATE ai_requests SET status = %s, response = %s, error = %s, took_ms = %s,
                                      usage = %s, input_tokens = %s, output_tokens = %s,
                                      total_tokens = %s, updated_at = now()
               WHERE id = %s""",
            (
                status,
                Jsonb(response) if response is not None else None,
                error,
                round((time.monotonic() - started) * 1000),
                Jsonb(said.usage) if said.usage is not None else None,
                said.input_tokens,
                said.output_tokens,
                said.total_tokens,
                row,
            ),
        )
