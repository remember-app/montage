import json
import logging
import time

import psycopg
from psycopg.types.json import Jsonb

from src.integrations import ai

log = logging.getLogger("integrations.ai_usage")

RATES_KEY = "ai.providerRates"


def _numbers(rate: object) -> tuple[float, float, float] | None:
    if not isinstance(rate, dict):
        return None
    try:
        return (
            float(rate["inputPerMillionUsd"]),
            float(rate["outputPerMillionUsd"]),
            float(rate["cachedInputPerMillionUsd"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _rate(conn: psycopg.Connection, model: str) -> tuple[float, float, float] | None:
    row = conn.execute("select value from settings where key = %s", (RATES_KEY,)).fetchone()
    if row is None or not isinstance(row[0], dict):
        return None
    return _numbers(row[0].get(model))


def _cost(rate: tuple[float, float, float] | None, usage: ai.Usage | None) -> float | None:
    if rate is None or usage is None:
        return None
    inputs, outputs = usage.input_tokens, usage.output_tokens
    if not isinstance(inputs, int) or not isinstance(outputs, int):
        return None
    cached = usage.cached_input_tokens or 0
    thought = usage.reasoning_tokens or 0
    uncached = max(0, inputs - cached)
    return (uncached * rate[0] + (outputs + thought) * rate[1] + cached * rate[2]) / 1_000_000


def _payload(
    conn: psycopg.Connection, model: str, usage: ai.Usage | None, duration: int, error: str | None
) -> dict:
    return {
        "tokens": {
            "inputTokens": usage.input_tokens if usage else None,
            "outputTokens": usage.output_tokens if usage else None,
            "totalTokens": usage.total_tokens if usage else None,
            "cachedTokens": usage.cached_input_tokens if usage else None,
            "reasoningTokens": usage.reasoning_tokens if usage else None,
        },
        "duration": duration,
        "costUsd": _cost(_rate(conn, model), usage),
        "error": error[:500] if error else None,
    }


def record(
    conn: psycopg.Connection,
    *,
    operation: str,
    status: str,
    provider: str,
    model: str,
    usage: ai.Usage | None,
    duration: int,
    error: str | None,
    job_id: int,
) -> None:
    conn.execute(
        """
        insert into ai_requests (job_id, operation, provider, model, status, usage)
        values (%s, %s, %s, %s, %s, %s)
        """,
        (
            job_id,
            operation,
            provider,
            model,
            status,
            Jsonb(_payload(conn, model, usage, duration, error)),
        ),
    )


def _json(reply: ai.Reply) -> dict:
    try:
        parsed = json.loads(reply.text)
    except ValueError:
        raise ai.AnswerError(
            f"the model did not answer with JSON: {reply.text[:500]}", reply.usage
        ) from None
    if not isinstance(parsed, dict):
        raise ai.AnswerError(
            f"the model answered with {type(parsed).__name__}, not an object", reply.usage
        )
    return parsed


def _recorded(conn: psycopg.Connection, *, operation: str, job_id: int, call, parse=None):
    chosen = ai.provider()
    started = time.monotonic()
    try:
        reply = call()
        result = parse(reply) if parse is not None else reply.text
    except RuntimeError as exc:
        record(
            conn,
            operation=operation,
            status="failed",
            provider=chosen.name,
            model=chosen.model,
            usage=getattr(exc, "usage", None),
            duration=int((time.monotonic() - started) * 1000),
            error=str(exc),
            job_id=job_id,
        )
        raise
    record(
        conn,
        operation=operation,
        status="ok",
        provider=chosen.name,
        model=chosen.model,
        usage=reply.usage,
        duration=int((time.monotonic() - started) * 1000),
        error=None,
        job_id=job_id,
    )
    return result, reply


def text(
    conn: psycopg.Connection,
    *,
    operation: str,
    instruction: str,
    prompt: str,
    blobs: tuple[ai.Blob, ...] = (),
    job_id: int,
) -> str:
    answer, _ = _recorded(
        conn, operation=operation, job_id=job_id, call=lambda: ai.ask(instruction, prompt, blobs)
    )
    return answer


def as_json(
    conn: psycopg.Connection,
    *,
    operation: str,
    instruction: str,
    prompt: str,
    blobs: tuple[ai.Blob, ...] = (),
    job_id: int,
) -> dict:
    parsed, _ = _recorded(
        conn,
        operation=operation,
        job_id=job_id,
        call=lambda: ai.ask(instruction, prompt, blobs),
        parse=_json,
    )
    return parsed


def reply(
    conn: psycopg.Connection,
    *,
    operation: str,
    instruction: str,
    prompt: str,
    blobs: tuple[ai.Blob, ...] = (),
    previous_interaction_id: str | None = None,
    job_id: int,
) -> tuple[str, str | None]:
    answer, response = _recorded(
        conn,
        operation=operation,
        job_id=job_id,
        call=lambda: ai.ask(
            instruction, prompt, blobs, previous_interaction_id=previous_interaction_id
        ),
    )
    return answer, response.interaction_id


def reply_json(
    conn: psycopg.Connection,
    *,
    operation: str,
    instruction: str,
    prompt: str,
    blobs: tuple[ai.Blob, ...] = (),
    previous_interaction_id: str | None = None,
    job_id: int,
) -> tuple[dict, str | None]:
    parsed, response = _recorded(
        conn,
        operation=operation,
        job_id=job_id,
        call=lambda: ai.ask(
            instruction, prompt, blobs, previous_interaction_id=previous_interaction_id
        ),
        parse=_json,
    )
    return parsed, response.interaction_id


if __name__ == "__main__":

    class Conn:
        def execute(self, *_):
            return self

        def fetchone(self):
            return (
                {
                    "m": {
                        "inputPerMillionUsd": 1,
                        "outputPerMillionUsd": 2,
                        "cachedInputPerMillionUsd": 0.1,
                    }
                },
            )

    rate = _rate(Conn(), "m")
    assert rate == (1.0, 2.0, 0.1)
    assert _cost(rate, ai.Usage(input_tokens=1_000_000, output_tokens=1_000_000)) == 3.0
    cached = ai.Usage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=1_000_000)
    assert _cost(rate, cached) == 0.1
    assert _cost(rate, None) is None
    print("ok")
