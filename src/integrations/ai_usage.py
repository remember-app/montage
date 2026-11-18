import logging
import time

import psycopg
from psycopg.types.json import Jsonb

from src import config
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


def _cost(rate: tuple[float, float, float] | None, usage: object) -> float | None:
    if rate is None or not isinstance(usage, dict):
        return None
    inputs = usage.get("total_input_tokens")
    outputs = usage.get("total_output_tokens")
    if not isinstance(inputs, (int, float)) or not isinstance(outputs, (int, float)):
        return None
    cached = usage.get("total_cached_tokens")
    thought = usage.get("total_thought_tokens")
    cached_tokens = cached if isinstance(cached, (int, float)) else 0
    thought_tokens = thought if isinstance(thought, (int, float)) else 0
    uncached = max(0, inputs - cached_tokens)
    return (
        uncached * rate[0] + (outputs + thought_tokens) * rate[1] + cached_tokens * rate[2]
    ) / 1_000_000


def _token(usage: object, name: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(name)
    return int(value) if isinstance(value, (int, float)) else None


def _payload(
    conn: psycopg.Connection,
    model: str,
    usage: object,
    duration: int,
    error: str | None,
) -> dict:
    return {
        "tokens": {
            "inputTokens": _token(usage, "total_input_tokens"),
            "outputTokens": _token(usage, "total_output_tokens"),
            "totalTokens": _token(usage, "total_tokens"),
            "cachedTokens": _token(usage, "total_cached_tokens"),
            "reasoningTokens": _token(usage, "total_thought_tokens"),
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
    usage: object,
    duration: int,
    error: str | None,
    job_id: int,
) -> None:
    model = config.gemini_model()
    conn.execute(
        """
        insert into ai_requests (job_id, operation, provider, model, status, usage)
        values (%s, %s, 'gemini', %s, %s, %s)
        """,
        (job_id, operation, model, status, Jsonb(_payload(conn, model, usage, duration, error))),
    )


def text(
    conn: psycopg.Connection,
    *,
    operation: str,
    instruction: str,
    prompt: str,
    blobs: tuple[ai.Blob, ...] = (),
    job_id: int,
) -> str:
    started = time.monotonic()
    try:
        answer, usage = ai.ask(instruction, prompt, blobs)
    except RuntimeError as exc:
        record(
            conn,
            operation=operation,
            status="failed",
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
        usage=usage,
        duration=int((time.monotonic() - started) * 1000),
        error=None,
        job_id=job_id,
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
    started = time.monotonic()
    try:
        parsed, usage = ai.as_json(instruction, prompt, blobs)
    except RuntimeError as exc:
        record(
            conn,
            operation=operation,
            status="failed",
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
        usage=usage,
        duration=int((time.monotonic() - started) * 1000),
        error=None,
        job_id=job_id,
    )
    return parsed
