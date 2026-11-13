import json
import logging
import uuid

import psycopg

from src.integrations import ai_usage
from src.movie import editor
from src.prompts.director import SYSTEM

log = logging.getLogger("movie.directing")


def _answer(answer: str) -> tuple[str, list]:
    try:
        parsed = json.loads(answer)
    except ValueError:
        raise RuntimeError(f"the director did not answer with JSON: {answer[:500]}") from None
    commands = parsed.get("commands") if isinstance(parsed, dict) else parsed
    if not isinstance(commands, list) or not commands:
        raise RuntimeError("the director answered without a non-empty commands list")
    summary = parsed.get("summary") if isinstance(parsed, dict) else None
    return summary if isinstance(summary, str) else "", commands


def _used(commands: list) -> list[str]:
    seen: list[str] = []
    for command in commands:
        if command.get("type") != "shot":
            continue
        item_id = str(command.get("itemId"))
        if item_id not in seen:
            seen.append(item_id)
    return seen


def _store(conn: psycopg.Connection, montage_id: uuid.UUID, commands: list, summary: str) -> None:
    landed = conn.execute(
        """
        update montages
        set manifest = jsonb_set(
                manifest, '{editingParams}', jsonb_build_object('commands', %s::jsonb)
            ),
            journal = jsonb_build_object(
                'directing', jsonb_build_object('text', %s::text, 'usedItemIds', %s::jsonb)
            )
        where id = %s
        """,
        (json.dumps(commands), summary, json.dumps(_used(commands)), montage_id),
    ).rowcount
    if not landed:
        raise RuntimeError(f"montage {montage_id} went away while directing")


def direct(
    conn: psycopg.Connection, montage_id: uuid.UUID, data: dict | str, attempts: int = 2
) -> list:
    report = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    problem = None
    for _ in range(attempts):
        prompt = report
        if problem is not None:
            prompt = (
                f"{report}\n\nYour previous answer was rejected: {problem}\n"
                "Fix the plan and answer with the schema again."
            )
        try:
            answer = ai_usage.text(
                conn, operation="text", montage_id=montage_id, instruction=SYSTEM, prompt=prompt
            )
            summary, commands = _answer(answer)
            editor.validate(conn, montage_id, commands)
        except RuntimeError as exc:
            problem = str(exc)
            log.warning("director answer for %s rejected: %s", montage_id, problem)
            continue
        _store(conn, montage_id, commands, summary)
        log.info("montage %s directed: %s commands", montage_id, len(commands))
        return commands
    raise RuntimeError(f"the director did not produce a valid plan: {problem}")
