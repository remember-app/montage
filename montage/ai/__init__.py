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
    def __init__(self, conn, provider=None):
        self.conn = conn
        self.provider = provider or from_env()

    def describe_media(self, media, cast=(), ref=None):
        known = ", ".join(cast) if cast else "пока никого"
        prompt = f"Уже известные действующие лица этого воспоминания: {known}."
        return self._call(
            "describe_media",
            DESCRIBE,
            prompt,
            media,
            ref,
            lambda provider: provider.generate(DESCRIBE, prompt, media, least_thinking=True),
            answers.as_json,
        )

    def transcribe_speech(self, audio, ref=None):
        return self._call(
            "transcribe_speech",
            TRANSCRIBE,
            "",
            [audio],
            ref,
            lambda provider: provider.transcribe(audio, least_thinking=True),
        )

    def plan_montage(
        self, shots, story="", target_seconds=None, music=(), ref=None, choose_music=False
    ):
        prompt = json.dumps(
            {
                "story": story,
                "target_seconds": target_seconds,
                "music": list(music),
                "shots": list(shots),
                **({"choose_music": True} if choose_music else {}),
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

    def _call(self, operation, instruction, prompt, blobs, ref, ask, read=None):
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
