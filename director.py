"""Режиссёр: in what order the material makes a film, how long each piece is held and what for.

The third module of the montage beside отбор and сборка (doc-03, «Сервис монтажа собран из отдельных
модулей»). It is handed the same judged pieces `selection` works on and, instead of scoring them,
asks the cloud model to direct them: it has read what is in every frame (`understanding.py`) and
what is said in every clip (`speech.py`), and it is given the organizer's own story, the two lists
and the length that was ordered. What comes back is an order, a length per piece and a note on what
each is there for.

The pieces it hands back are `assembly`'s to lay down, exactly as `selection.choose`'s are — nothing
downstream knows which of the two cut the video. And `None` is an answer here: no cloud model
configured, nothing described yet, or a plan with nothing of ours in it. On `None` the montage is
cut the way every montage in production is cut today (`selection.choose`), which is the whole state
of the worker while there is no key (decision-10).

The organizer beats the director exactly as it beats the algorithm, and for the same reason it is
written in one place: a struck-out piece is never *shown* to the model — the surest way for it not
to come back — and a piece marked «включить» is in the film whether the plan asked for it or not.
"""

import dataclasses
import logging

import ai
import assembly
import selection

# What one piece may be held for, whatever the plan says. Under a second is a flash nobody sees;
# longer than the ceiling is a photograph that reads as a frozen video. A clip is bounded by itself
# as well — `selection.cut_to` never asks a file for more than it has.
MIN_PIECE_SECONDS = 1.0
MAX_PIECE_SECONDS = 8.0

# How much of one transcript goes into the prompt. The director needs to know what is said in a
# clip, not to be handed ten minutes of it: a whole memory travels in one request, and every word
# of it is paid for.
SPEECH_CHARACTERS = 400

log = logging.getLogger("montage.director")


def direct(conn, pieces, target_seconds, include=(), exclude=(), story="", ref=None, brain=None):
    """The film the model plans out of these pieces — or None, and then nobody directed anything.

    `pieces` are judged (`selection.judge`), `include`/`exclude` are the organizer's two lists from
    the order and `story` is what they said the video is about. `brain` is for the tests; the worker
    takes the gateway the environment gives it.
    """
    brain = brain or ai.AI(conn)
    if brain.provider is None:
        # Today's state, and the one thing that must not change because of this module: with nobody
        # to ask, the montage is the montage it has always been.
        log.info("no cloud model: montage %s is cut by the worker's own choosing", ref)
        return None

    include, exclude = {str(one) for one in include}, {str(one) for one in exclude}
    material = list(_material(pieces, include, exclude))
    # One shot per moment, by the отбор's own дедуп: a director shown a whole burst can put two
    # frames of the same second next to each other in the plan, and every duplicate in the question
    # is paid for as well. «Включить» wins its burst here exactly as it does in `selection.choose`.
    _, duplicates = selection._dedup(material, always=include)
    swallowed = {piece.media_id for piece in duplicates}
    candidates = {piece.media_id: piece for piece in material if piece.media_id not in swallowed}
    known = _known(conn, candidates)
    if not any(what.get("description") for what in known.values()):
        # Nothing has been described yet — the memory became ready before there was a key, and the
        # hourly sweep has not caught up. A director shown nothing but file kinds would be ordering
        # frames it has never seen, for money; the algorithm at least looks at the pixels.
        log.info("montage %s: nothing in this memory is described yet, leaving it to the worker's "
                 "own choosing", ref)
        return None

    planned = brain.plan_montage(
        [_shot(piece, known, include) for piece in candidates.values()],
        story=story, target_seconds=target_seconds, ref=ref,
    )
    log.info("montage %s: the director's plan is %s", ref, planned)

    chosen, taken = [], set()
    for step in (planned or {}).get("plan") or []:
        piece = candidates.get(str(step.get("media_id", "")))
        if piece is None or piece.media_id in taken:
            # A piece the model invented, one it was never shown, or the same one twice. The plan is
            # the model's; what material there is stays ours.
            log.info("montage %s: the plan names %s, which is not material of ours or is already "
                     "in the film — skipped", ref, step.get("media_id"))
            continue
        taken.add(piece.media_id)
        chosen.append(_cut(piece, step.get("seconds")))

    # «Включить» is not the model's to refuse. A piece the organizer named and the plan left out
    # goes in at the end of it: the order is the director's, and there is no place inside it that is
    # obviously this piece's own.
    for media_id, piece in candidates.items():
        if media_id in include and media_id not in taken:
            log.info("montage %s: the plan left out %s, which the order asks for — putting it back",
                     ref, media_id)
            chosen.append(_cut(piece, None))

    fitted = []
    for piece in chosen:
        # The length that was ordered, held to here rather than trusted to the prompt — except over
        # «включить», which is not weighed against anything, and over the first piece, because a
        # montage shorter than asked for is a montage and an empty one is not.
        if fitted and piece.media_id not in include \
                and assembly.length_of(fitted + [piece]) > target_seconds:
            log.info("montage %s: the plan runs past the %.0fs ordered, %s is left out of it",
                     ref, target_seconds, piece.media_id)
            continue
        fitted.append(piece)
    if not fitted:
        log.warning("montage %s: the plan had nothing of ours in it, cutting the video by the "
                    "worker's own choosing instead", ref)
        return None
    log.info("montage %s is cut to a plan of %s piece(s), %.1fs", ref, len(fitted),
             assembly.length_of(fitted))
    return fitted


def _material(pieces, include, exclude):
    """What the model is allowed to build out of, in the order the memory is read in.

    Struck-out material is not filtered out of the answer, it never goes into the question. A file
    with no frame in it at all is out for the reason `selection.choose` keeps it out: there is no
    picture to put on screen and «включить» cannot conjure one. What отбор merely disliked — blur,
    exposure — is out unless the organizer asked for it by name, so the director is offered what a
    person would be offered. A burst of shots of one moment is thinned to one by `direct`
    afterwards, for the same reason: the offer is what a person would look through, not the whole
    camera roll.
    """
    for piece in pieces:
        if piece.media_id in exclude or piece.rejected == selection.UNREADABLE:
            continue
        if piece.rejected and piece.media_id not in include:
            continue
        yield piece


def _known(conn, media_ids):
    """What the pipeline has already learned about these pieces, out of its own two tables.

    Nothing is asked of a model here: the description of a frame and the words in a clip are written
    once and read ever after (doc-03), and the director is the reader they were written for. A piece
    nobody has described yet simply has nothing under it — it is offered to the director as a file
    and not as a scene.
    """
    return {
        media_id: {
            name: value
            for name, value in (
                ("description", description or ""),
                ("cast", cast or []),
                ("speech", (transcript or "")[:SPEECH_CHARACTERS]),
            )
            if value
        }
        for media_id, description, cast, transcript in conn.execute(
            """SELECT mine.id, u.description, u.cast_labels, s.transcript
               FROM unnest(%s::text[]) AS mine(id)
               LEFT JOIN media_understandings u ON u.media_id = mine.id
               LEFT JOIN media_speech s ON s.media_id = mine.id""",
            (list(media_ids),),
        ).fetchall()
    }


def _shot(piece, known, include):
    """One piece as the model is shown it: what is in it, and what may be done with it."""
    return {
        "media_id": piece.media_id,
        "kind": piece.kind,
        # What this file can actually give. A clip cannot be held longer than it runs, and a plan
        # that asked for it anyway would be quietly cut back to the truth by `_cut`.
        "seconds_available": round(_available(piece), 1),
        **known.get(piece.media_id, {}),
        **({"must_include": True} if piece.media_id in include else {}),
    }


def _available(piece):
    return min(MAX_PIECE_SECONDS, piece.duration) if piece.duration else MAX_PIECE_SECONDS


def _cut(piece, seconds):
    """The piece at the length the plan gave it — within what the screen and the file allow.

    A step that says nothing about its length, or says something that is not a number, gets the
    length отбор cut it at: the plan is about the order of the film first, and a missing number is
    not a reason to leave a piece out of it. `NaN` and `Infinity` need no guard of their own here:
    they would walk straight through `min`/`max` — every comparison against them is false — but
    `ai._as_json` refuses an answer carrying either, so no plan holding one ever gets this far.
    """
    try:
        seconds = min(max(float(seconds), MIN_PIECE_SECONDS), MAX_PIECE_SECONDS)
    except (TypeError, ValueError):
        seconds = piece.seconds
    # `rejected` goes with it: the piece is in the film, so nothing about it is turned down — the
    # same thing `selection.choose` does to a piece the organizer overruled it on.
    return dataclasses.replace(selection.cut_to(piece, seconds), rejected=None)
