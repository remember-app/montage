import json
import logging
import uuid

import psycopg

from src.integrations import ai
from src.movie import editor

log = logging.getLogger("movie.directing")

SYSTEM = """
You are the director of a short movie made from the material of one event. You receive a report
about the event: the order settings, the shots with their descriptions and analyses, and the
available music. You decide what goes into the movie, in what order, how long each shot runs, and
how the music plays.

Answer in English. Reply with a JSON object only, in this shape:
{"commands": [ ... ]}

A shot command is:
{"type": "shot", "itemId": "...", "mediaId": "...", "kind": "photo | video | audio",
 "sourceStartSeconds": 0, "seconds": 2.5, "liveLevel": 0, "framing": null,
 "motion": "none | zoom_in | zoom_out", "transitionSeconds": 0.5}

A music command is:
{"type": "music", "songId": "...", "mediaId": "...", "sourceStartSeconds": 0, "seconds": 6,
 "atSeconds": 0, "gain": 1.0, "fadeInSeconds": 0.5, "fadeOutSeconds": 0.5}

Rules:
- Shots play in the order you give them. Every shot's itemId, mediaId and kind must be copied
  exactly from the report. Never invent an item.
- seconds is the time on screen, sourceStartSeconds is the position in the source file.
- Photos use sourceStartSeconds 0. Video and audio must stay inside the file.
- Audio shots are shown with their cover image; their own sound is the live sound.
- liveLevel is 0..1: 0 mutes the live sound of the shot, 1 keeps it. Photos are always 0.
- framing is null for the whole frame, or {x, y, w, h} as fractions of the frame.
- transitionSeconds is the overlap with the next shot; 0 means a cut.
- Music is any number of cues. Each cue takes a segment from any song in the report; atSeconds is
  its place on the movie timeline. gain is 0..2, 1 is the recorded level. Cues may overlap.
- Every required item must appear. Use only items from the report.
- The total movie should be close to targetSeconds. The plan is the whole movie.
- Do not add fields outside the schema. Do not explain. JSON only.
""".strip()


def _commands(answer: str) -> list:
    try:
        parsed = json.loads(answer)
    except ValueError:
        raise RuntimeError(f"the director did not answer with JSON: {answer[:500]}") from None
    commands = parsed.get("commands") if isinstance(parsed, dict) else parsed
    if not isinstance(commands, list) or not commands:
        raise RuntimeError("the director answered without a non-empty commands list")
    return commands


def _store(conn: psycopg.Connection, montage_id: uuid.UUID, commands: list) -> None:
    landed = conn.execute(
        """
        update montages
        set manifest = jsonb_set(
            manifest, '{editingParams}', jsonb_build_object('commands', %s::jsonb)
        )
        where id = %s
        """,
        (json.dumps(commands), montage_id),
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
            commands = _commands(ai.ask(SYSTEM, prompt))
            editor.validate(conn, montage_id, commands)
        except RuntimeError as exc:
            problem = str(exc)
            log.warning("director answer for %s rejected: %s", montage_id, problem)
            continue
        _store(conn, montage_id, commands)
        log.info("montage %s directed: %s commands", montage_id, len(commands))
        return commands
    raise RuntimeError(f"the director did not produce a valid plan: {problem}")
