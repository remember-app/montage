import logging
import time

import psycopg

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


def record(
    conn: psycopg.Connection,
    *,
    operation: str,
    status: str,
    usage: object,
    duration: int,
    error: str | None,
    media_id=None,
    montage_id=None,
) -> None:
    model = config.gemini_model()
    conn.execute(
        """
        insert into ai_requests (
            media_id, montage_id, operation, provider, model, status,
            input_tokens, output_tokens, total_tokens, cached_tokens, reasoning_tokens,
            duration, cost_usd, error
        )
        values (%s, %s, %s, 'gemini', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            media_id,
            montage_id,
            operation,
            model,
            status,
            _token(usage, "total_input_tokens"),
            _token(usage, "total_output_tokens"),
            _token(usage, "total_tokens"),
            _token(usage, "total_cached_tokens"),
            _token(usage, "total_thought_tokens"),
            duration,
            _cost(_rate(conn, model), usage),
            error[:500] if error else None,
        ),
    )


def text(
    conn: psycopg.Connection,
    *,
    operation: str,
    instruction: str,
    prompt: str,
    blobs: tuple[ai.Blob, ...] = (),
    media_id=None,
    montage_id=None,
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
            media_id=media_id,
            montage_id=montage_id,
        )
        raise
    record(
        conn,
        operation=operation,
        status="ok",
        usage=usage,
        duration=int((time.monotonic() - started) * 1000),
        error=None,
        media_id=media_id,
        montage_id=montage_id,
    )
    return answer


def as_json(
    conn: psycopg.Connection,
    *,
    operation: str,
    instruction: str,
    prompt: str,
    blobs: tuple[ai.Blob, ...] = (),
    media_id=None,
    montage_id=None,
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
            media_id=media_id,
            montage_id=montage_id,
        )
        raise
    record(
        conn,
        operation=operation,
        status="ok",
        usage=usage,
        duration=int((time.monotonic() - started) * 1000),
        error=None,
        media_id=media_id,
        montage_id=montage_id,
    )
    return parsed
