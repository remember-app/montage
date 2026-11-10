import json
import logging
import uuid

import psycopg

from src import config
from src.integrations import ai_usage
from src.movie import editor

log = logging.getLogger("movie.directing")

SYSTEM = """
You are the director of a short movie made from the material of one event. You receive a report
about the event: the order settings, the shots with their descriptions and analyses, and the
available music. You decide what goes into the movie, in what order, how long each shot runs, and
how the music plays.

Answer in English. Reply with a JSON object only, in this shape:
{"summary": "...", "commands": [ ... ]}

summary is one short paragraph for the participant who ordered the movie: how the movie was put
together, which shots made it in, how the pacing and the music work. It is not a justification of
each command and not a list of rules.

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
- Do not add fields outside the schema. No explanation outside summary. JSON only.
""".strip()


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


def _mock_plan(report: dict) -> list:
    commands = []
    for shot in report.get("shots") or []:
        kind = shot.get("kind")
        seconds = 2.5 if kind == "photo" else min(3.0, float(shot.get("durationSeconds") or 3.0))
        commands.append(
            {
                "type": "shot",
                "itemId": shot["itemId"],
                "mediaId": shot["mediaId"],
                "kind": kind,
                "sourceStartSeconds": 0.0,
                "seconds": round(max(seconds, 0.5), 3),
                "liveLevel": 0.0 if kind == "photo" else 0.6,
                "framing": None,
                "motion": "none",
                "transitionSeconds": 0.25,
            }
        )
    total = sum(command["seconds"] for command in commands) - 0.25 * max(len(commands) - 1, 0)
    tracks = report.get("music") or []
    if tracks and total > 0:
        track = tracks[0]
        seconds = min(float(track.get("durationSeconds") or total), total)
        fade = min(0.5, seconds)
        commands.append(
            {
                "type": "music",
                "songId": track["songId"],
                "mediaId": track["mediaId"],
                "sourceStartSeconds": 0.0,
                "seconds": round(seconds, 3),
                "atSeconds": 0.0,
                "gain": 0.8,
                "fadeInSeconds": fade,
                "fadeOutSeconds": fade,
            }
        )
    return commands


def direct(
    conn: psycopg.Connection, montage_id: uuid.UUID, data: dict | str, attempts: int = 2
) -> list:
    if config.ai_mock():
        parsed = data if isinstance(data, dict) else json.loads(data)
        commands = _mock_plan(parsed)
        music = any(one.get("type") == "music" for one in commands)
        summary = (
            f"Mock edit: {len(_used(commands))} shots, {'with music' if music else 'no music'}."
        )
        editor.validate(conn, montage_id, commands)
        _store(conn, montage_id, commands, summary)
        log.info("montage %s directed in mock mode: %s commands", montage_id, len(commands))
        return commands
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
