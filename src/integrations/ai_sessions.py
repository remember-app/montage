import datetime
import json
import logging
import time

import psycopg
from psycopg.types.json import Jsonb

from src.integrations import ai

log = logging.getLogger("integrations.ai_sessions")

RATES_KEY = "ai.providerRates"
TOKEN_KEYS = ("inputTokens", "outputTokens", "totalTokens", "cachedTokens", "reasoningTokens")


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


def _tokens(usage: ai.Usage | None) -> dict:
    return {
        "inputTokens": usage.input_tokens if usage else None,
        "outputTokens": usage.output_tokens if usage else None,
        "totalTokens": usage.total_tokens if usage else None,
        "cachedTokens": usage.cached_input_tokens if usage else None,
        "reasoningTokens": usage.reasoning_tokens if usage else None,
    }


def _sum(first, second):
    if first is None:
        return second
    if second is None:
        return first
    return first + second


def _usage(conn: psycopg.Connection, model: str, usage: ai.Usage | None, duration: int) -> dict:
    return {
        "tokens": _tokens(usage),
        "duration": duration,
        "costUsd": _cost(_rate(conn, model), usage),
    }


def _merged(
    previous: dict, conn: psycopg.Connection, model: str, usage: ai.Usage | None, duration: int
) -> dict:
    current = _usage(conn, model, usage, duration)
    previous_tokens = previous.get("tokens") or {}
    return {
        "tokens": {
            key: _sum(previous_tokens.get(key), current["tokens"].get(key)) for key in TOKEN_KEYS
        },
        "duration": _sum(previous.get("duration"), current["duration"]),
        "costUsd": _sum(previous.get("costUsd"), current["costUsd"]),
    }


def _entry(
    instruction: str,
    prompt: str,
    blobs: tuple[ai.Blob, ...],
    tool_results: tuple[ai.ToolResult, ...],
    reply: ai.Reply | None,
    previous_interaction_id: str | None,
) -> dict:
    text = (
        prompt
        if previous_interaction_id
        else "\n\n".join(part for part in (instruction, prompt) if part)
    )
    return {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input": {
            "text": text,
            "blobs": [blob.mime_type for blob in blobs],
            "toolResults": [
                {
                    "callId": result.call_id,
                    "name": result.name,
                    "text": result.text,
                    "isError": result.is_error,
                }
                for result in tool_results
            ],
        },
        "output": {
            "text": reply.text if reply else "",
            "toolCalls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in (reply.tool_calls if reply else ())
            ],
            "interactionId": reply.interaction_id if reply else None,
        },
    }


def record(
    conn: psycopg.Connection,
    *,
    session_id: int | None,
    operation: str,
    status: str,
    provider: str,
    model: str,
    usage: ai.Usage | None,
    duration: int,
    error: str | None,
    entry: dict,
    job_id: int,
) -> int:
    if session_id is None:
        row = conn.execute(
            """
            insert into ai_sessions
                (job_id, operation, provider, model, status, usage, error, transcript)
            values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            returning id
            """,
            (
                job_id,
                operation,
                provider,
                model,
                status,
                Jsonb(_usage(conn, model, usage, duration)),
                error[:500] if error else None,
                Jsonb([entry]),
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("the ai session was not created")
        return int(row[0])

    previous = conn.execute("select usage from ai_sessions where id = %s", (session_id,)).fetchone()
    if previous is None:
        raise RuntimeError(f"ai session {session_id} is gone")
    row = conn.execute(
        """
        update ai_sessions
        set status = %s,
            error = %s,
            usage = %s,
            transcript = coalesce(transcript, '[]'::jsonb) || %s::jsonb,
            updated_at = now()
        where id = %s
        returning id
        """,
        (
            status,
            error[:500] if error else None,
            Jsonb(_merged(previous[0] or {}, conn, model, usage, duration)),
            Jsonb([entry]),
            session_id,
        ),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"ai session {session_id} is gone")
    return int(row[0])


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


def turn(
    conn: psycopg.Connection,
    *,
    operation: str,
    instruction: str,
    prompt: str,
    blobs: tuple[ai.Blob, ...] = (),
    previous_interaction_id: str | None = None,
    tools: tuple[ai.Tool, ...] = (),
    tool_results: tuple[ai.ToolResult, ...] = (),
    session_id: int | None = None,
    background: bool = False,
    stream: bool = False,
    job_id: int,
) -> tuple[ai.Reply, int]:
    chosen = ai.provider()
    started = time.monotonic()
    try:
        reply = ai.ask(
            instruction,
            prompt,
            blobs,
            previous_interaction_id=previous_interaction_id,
            tools=tools,
            tool_results=tool_results,
            background=background,
            stream=stream,
        )
    except RuntimeError as exc:
        record(
            conn,
            session_id=session_id,
            operation=operation,
            status="failed",
            provider=chosen.name,
            model=chosen.model,
            usage=getattr(exc, "usage", None),
            duration=int((time.monotonic() - started) * 1000),
            error=str(exc),
            entry=_entry(instruction, prompt, blobs, tool_results, None, previous_interaction_id),
            job_id=job_id,
        )
        raise
    session = record(
        conn,
        session_id=session_id,
        operation=operation,
        status="ok",
        provider=chosen.name,
        model=chosen.model,
        usage=reply.usage,
        duration=int((time.monotonic() - started) * 1000),
        error=None,
        entry=_entry(instruction, prompt, blobs, tool_results, reply, previous_interaction_id),
        job_id=job_id,
    )
    return reply, session


def text(
    conn: psycopg.Connection,
    *,
    operation: str,
    instruction: str,
    prompt: str,
    blobs: tuple[ai.Blob, ...] = (),
    job_id: int,
) -> str:
    reply, _ = turn(
        conn,
        operation=operation,
        instruction=instruction,
        prompt=prompt,
        blobs=blobs,
        job_id=job_id,
    )
    return reply.text


def as_json(
    conn: psycopg.Connection,
    *,
    operation: str,
    instruction: str,
    prompt: str,
    blobs: tuple[ai.Blob, ...] = (),
    job_id: int,
) -> dict:
    reply, _ = turn(
        conn,
        operation=operation,
        instruction=instruction,
        prompt=prompt,
        blobs=blobs,
        job_id=job_id,
    )
    return _json(reply)


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
    answer, _ = turn(
        conn,
        operation=operation,
        instruction=instruction,
        prompt=prompt,
        blobs=blobs,
        previous_interaction_id=previous_interaction_id,
        job_id=job_id,
    )
    return answer.text, answer.interaction_id


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
    answer, _ = turn(
        conn,
        operation=operation,
        instruction=instruction,
        prompt=prompt,
        blobs=blobs,
        previous_interaction_id=previous_interaction_id,
        job_id=job_id,
    )
    return _json(answer), answer.interaction_id


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
    merged = _merged(
        {"tokens": {"inputTokens": 10}, "duration": 5, "costUsd": 0.2},
        Conn(),
        "m",
        ai.Usage(input_tokens=10, output_tokens=5),
        7,
    )
    assert merged["tokens"]["inputTokens"] == 20
    assert merged["duration"] == 12
    assert merged["costUsd"] is not None
    print("ok")
