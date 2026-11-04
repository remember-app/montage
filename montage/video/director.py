import dataclasses
import logging

from montage import ai
from montage.media import understanding
from montage.video import assembly, selection

MIN_PIECE_SECONDS = 1.0
MAX_PIECE_SECONDS = 8.0

SPEECH_CHARACTERS = 400


log = logging.getLogger("montage.director")


def direct(
    conn,
    pieces,
    target_seconds,
    exclude=(),
    story="",
    ref=None,
    brain=None,
    music=(),
    choose_music=False,
):
    brain = brain or ai.AI(conn)
    if brain.provider is None:
        log.info("no cloud model: montage %s is cut by the worker's own choosing", ref)
        return None, (), None

    candidates = {piece.media_id: piece for piece in _material(pieces, exclude)}
    known = _known(conn, candidates)
    if not any(what.get("description") for what in known.values()):
        log.info(
            "montage %s: nothing in this memory is described yet, leaving it to the worker's "
            "own choosing",
            ref,
        )
        return None, (), None

    planned = (
        brain.plan_montage(
            [_shot(piece, known) for piece in candidates.values()],
            story=story,
            target_seconds=target_seconds,
            music=music,
            ref=ref,
            choose_music=choose_music,
        )
        or {}
    )
    log.info("montage %s: the director's plan is %s", ref, planned)
    chosen_track = planned.get("track") if choose_music else None

    chosen, taken = [], set()
    for step in planned.get("plan") or []:
        piece = candidates.get(step["media_id"])
        if piece is not None:
            piece = dataclasses.replace(
                _cut(piece, step["seconds"], step["start"]), live=step["live"]
            )
            key = (
                (piece.media_id, piece.start, piece.seconds)
                if piece.kind == "video"
                else piece.media_id
            )
        if piece is None or key in taken:
            log.info(
                "montage %s: the plan names %s, which is not material of ours or is already "
                "in the film — skipped",
                ref,
                step["media_id"],
            )
            continue
        taken.add(key)
        chosen.append((piece, _thought(step)))

    in_plan = {piece.media_id for piece, _ in chosen}

    fitted = []
    for piece, thought in chosen:
        if fitted and assembly.length_of(fitted + [piece]) > target_seconds:
            log.info(
                "montage %s: the plan runs past the %.0fs ordered, %s is left out of it",
                ref,
                target_seconds,
                piece.media_id,
            )
            thought.update(decision="skip", reason=selection.LEFT_OUT)
            continue
        thought["decision"] = "take"
        fitted.append(piece)

    thoughts = [{"thought": planned["overview"]}] if planned.get("overview") else []
    thoughts += (thought for _, thought in chosen)
    thoughts += (
        {
            "media_id": media_id,
            "decision": "skip",
            "reason": selection.LEFT_OUT,
            "thought": planned.get("left_out", {}).get(media_id),
        }
        for media_id in candidates
        if media_id not in in_plan
    )

    if not fitted:
        log.warning(
            "montage %s: the plan had nothing of ours in it, cutting the video by the "
            "worker's own choosing instead",
            ref,
        )
        return None, (), chosen_track
    log.info(
        "montage %s is cut to a plan of %s piece(s), %.1fs",
        ref,
        len(fitted),
        assembly.length_of(fitted),
    )
    return fitted, thoughts, chosen_track


def _material(pieces, exclude):
    for piece in pieces:
        if piece.media_id in exclude or piece.rejected == selection.UNREADABLE:
            continue
        yield piece


def _known(conn, media_ids):
    known = {
        media_id: {
            name: value
            for name, value in (
                ("description", description or ""),
                ("cast", cast or []),
                ("speech", (transcript or "")[:SPEECH_CHARACTERS]),
                *((details or {}).items()),
            )
            if value
        }
        for media_id, description, cast, transcript, details in conn.execute(
            """SELECT mine.id, u.description, u.cast_labels, s.transcript, u.details
               FROM unnest(%s::text[]) AS mine(id)
               LEFT JOIN media_understandings u ON u.media_id = mine.id
               LEFT JOIN media_speech s ON s.media_id = mine.id""",
            (list(media_ids),),
        ).fetchall()
    }
    for media_id, said in understanding.frame_comments(conn, list(media_ids)).items():
        if said:
            known.setdefault(media_id, {})["comments"] = said
    return known


def _shot(piece, known):
    return {
        "media_id": piece.media_id,
        "kind": piece.kind,
        "seconds_available": round(piece.duration or MAX_PIECE_SECONDS, 1),
        **known.get(piece.media_id, {}),
    }


def _thought(step):
    return {
        "media_id": step["media_id"],
        "thought": step["thought"],
        "story_part": step["part"],
        "story_part_title": step["part_title"],
        **{f"focus_{side}": part for side, part in (step["focus"] or {}).items()},
    }


def _cut(piece, seconds=None, start=None):
    if seconds is None:
        seconds = piece.seconds
    else:
        seconds = min(max(seconds, MIN_PIECE_SECONDS), MAX_PIECE_SECONDS)
    if start is not None and piece.kind == "video" and piece.duration:
        held = min(seconds, piece.duration)
        start = min(max(start, 0.0), piece.duration - held)
        piece = dataclasses.replace(piece, best_second=start + held / 2)
    return dataclasses.replace(selection.cut_to(piece, seconds), rejected=None)
