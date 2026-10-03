"""Режиссёр: in what order the material makes a film, how long each piece is held and what for.

The third module of the montage beside отбор and сборка (doc-03, «Сервис монтажа собран из отдельных
модулей»). It is handed the same prepared pieces `selection` works on and, instead of scoring them,
asks the cloud model to direct them: it has read what is in every frame (`understanding.py`) and
what is said in every clip (`speech.py`), and it is given the organizer's own story, the list of
frames left out by hand and the length that was ordered. What comes back is an order, a length per
piece — out of a clip, the stretch of it, and as many stretches of one clip as the film wants —
what each piece is heard with (the music, its own sound or the two mixed) and a note on what each is
there for.

The pieces it hands back are `assembly`'s to lay down, exactly as `selection.choose`'s are — nothing
downstream knows which of the two cut the video. And `(None, ())` is an answer here: no cloud model
configured, nothing described yet, or a plan with nothing of ours in it. On it the montage is cut
the way every montage in production is cut today (`selection.choose`), which is the whole state of
the worker while there is no key (decision-10).

Рядом с кусками едут слова, которыми режиссёр объяснил сам себя: мысль о плане в целом и по записи
на каждый показанный ему кадр — мысль, часть истории, рамка «куда смотрит», решение и причина, если
кадр остался за бортом. Собираются они здесь, потому что больше никто не знает, что модель видела и
что она об этом сказала, а в журнал их кладёт оркестратор (`montage.py`) — как и весь остальной
рассказ о сборке.

The organizer beats the director exactly as it beats the algorithm, and for the same reason it is
written in one place: a struck-out piece is never *shown* to the model — the surest way for it not
to come back. There is no overruling in the other direction: the frames left at the order are equal,
so the director picks out of all of them and nobody tells it which one matters more.
"""

import dataclasses
import logging

from montage import ai
from montage.video import assembly, selection

# What one piece may be held for, whatever the plan says. Under a second is a flash nobody sees;
# longer than the ceiling is a photograph that reads as a frozen video. A clip is bounded by itself
# as well — `selection.cut_to` never asks a file for more than it has. The prompt (`ai.PLAN`) names
# the model the same window, in words.
MIN_PIECE_SECONDS = 1.0
MAX_PIECE_SECONDS = 8.0

# How much of one transcript goes into the prompt. The director needs to know what is said in a
# clip, not to be handed ten minutes of it: a whole memory travels in one request, and every word
# of it is paid for.
SPEECH_CHARACTERS = 400

# Почему кадр остался за бортом, когда своих слов об этом у модели не нашлось: она просто не взяла
# его в план. Остальные коды словаря причин — отсева (`selection`), этот берётся только здесь.
LEFT_OUT = "left_out"

log = logging.getLogger("montage.director")


def direct(conn, pieces, target_seconds, exclude=(), story="", ref=None, brain=None, music=()):
    """The film the model plans out of these pieces and what it said while planning it — or
    `(None, ())`, and then nobody directed anything.

    `pieces` are prepared (`selection.judge`), `exclude` is the organizer's own list of media ids
    from the order and `story` is what they said the video is about. `music` is what will play under
    the video — the tracks in the order they play, how long each runs and, on a track the organizer
    wished something of, that wish in their own words (`music.heard`); the soundtrack itself is
    assembled from the order's list either way, so it is here to be read and not to be obeyed.
    `brain` is for the tests; the worker takes the gateway the environment gives it.

    Слова режиссёра — вторым, готовыми полями журнала (`journal.COLUMNS`), порядком плана: первой
    мысль о плане в целом, за ней по записи на каждый кусок плана — у видео их бывает несколько, —
    и «мимо» на каждом показанном модели кадре, которого в плане нет. «Беру» среди них стоят тем
    же порядком, что куски ролика.
    """
    brain = brain or ai.AI(conn)
    if brain.provider is None:
        # Today's state, and the one thing that must not change because of this module: with nobody
        # to ask, the montage is the montage it has always been.
        log.info("no cloud model: montage %s is cut by the worker's own choosing", ref)
        return None, ()

    # Порядок вопроса — наш: спрашивается тот же материал и в том же порядке, в каком читается
    # воспоминание. Режиссёру показывается всё, что осталось после заказа, серия одинаковых кадров
    # в том числе: свёртывать её за его спиной было бы тем же отсевом, только тише.
    candidates = {piece.media_id: piece for piece in _material(pieces, exclude)}
    known = _known(conn, candidates)
    if not any(what.get("description") for what in known.values()):
        # Nothing has been described yet — the memory became ready before there was a key, and the
        # hourly sweep has not caught up. A director shown nothing but file kinds would be ordering
        # frames it has never seen, for money; the algorithm at least looks at the pixels.
        log.info(
            "montage %s: nothing in this memory is described yet, leaving it to the worker's "
            "own choosing",
            ref,
        )
        return None, ()

    planned = (
        brain.plan_montage(
            [_shot(piece, known) for piece in candidates.values()],
            story=story,
            target_seconds=target_seconds,
            music=music,
            ref=ref,
        )
        or {}
    )
    log.info("montage %s: the director's plan is %s", ref, planned)

    # Кусок плана — вместе со словами режиссёра о нём: одно видео может стоять в плане несколькими
    # отрезками, и «беру» или «мимо» решается о каждом отрезке, а не о файле.
    chosen, taken = [], set()
    for step in planned.get("plan") or []:
        piece = candidates.get(step["media_id"])
        if piece is not None:
            # Звук — тоже слово режиссёра о куске: чем кусок звучит, решает он, а слышно ли
            # у куска вообще хоть что-то своё, знает только сборка (`assembly.assemble`).
            piece = dataclasses.replace(
                _cut(piece, step["seconds"], step["start"]), live=step["live"]
            )
            # A photograph has one stretch, the whole of it, whatever length it is held for.
            key = (
                (piece.media_id, piece.start, piece.seconds)
                if piece.kind == "video"
                else piece.media_id
            )
        if piece is None or key in taken:
            # A piece the model invented, one it was never shown, or the same stretch of the same
            # file twice. The plan is the model's; what material there is stays ours — и в журнале
            # ему тоже не место: рассказ идёт о кадрах воспоминания, а не о том, что модель себе
            # вообразила.
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
        # The length that was ordered, held to here rather than trusted to the prompt — and except
        # over the first piece, because a montage shorter than asked for is a montage and an empty
        # one is not. Every other piece is weighed like any: кадры заказа равны, и «включить» больше
        # ничего не закрепляет (doc-03, «What it doesn't»).
        # Решение ставится здесь, а не там, где сказаны слова: кусок, который не влез, — это
        # «мимо» с его же мыслью при нём.
        if fitted and assembly.length_of(fitted + [piece]) > target_seconds:
            log.info(
                "montage %s: the plan runs past the %.0fs ordered, %s is left out of it",
                ref,
                target_seconds,
                piece.media_id,
            )
            thought.update(decision="skip", reason=selection.NO_ROOM)
            continue
        thought["decision"] = "take"
        fitted.append(piece)

    # Мысль о плане в целом — первой записью этапа и без кадра: она про фильм, а не про кусок. За
    # ней — по записи на каждый кусок плана, его порядком: взятые идут тем же порядком, что и куски
    # ролика, и по нему журнал связывает слова с отрезком (`story.plan`).
    thoughts = [{"thought": planned["overview"]}] if planned.get("overview") else []
    thoughts += (thought for _, thought in chosen)
    # Остальное, что модели показали: «мимо» с кодом `left_out` и её словами о том, почему, —
    # мыслью: причину журнал держит кодом, и слова в ней приложение показало бы как
    # «не подошёл». Ни вычеркнутого, ни непрочитанного модель не видела: первый убрал сам
    # организатор, второй назван на подготовке, и в журнале их нет.
    thoughts += (
        {
            "media_id": media_id,
            "decision": "skip",
            "reason": LEFT_OUT,
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
        return None, ()
    log.info(
        "montage %s is cut to a plan of %s piece(s), %.1fs",
        ref,
        len(fitted),
        assembly.length_of(fitted),
    )
    return fitted, thoughts


def _material(pieces, exclude):
    """What the model is allowed to build out of, in the order the memory is read in.

    Struck-out material is not filtered out of the answer, it never goes into the question. A file
    with no frame in it at all is out for the reason `selection.choose` keeps it out: there is no
    picture to put on screen. Everything else goes in — смазанный, тёмный и повторяющиеся кадры
    серии в том числе: кадры, оставленные при выборе, равны, и решение по каждому — у режиссёра.
    """
    for piece in pieces:
        if piece.media_id in exclude or piece.rejected == selection.UNREADABLE:
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


def _shot(piece, known):
    """One piece as the model is shown it: what is in it, and what may be done with it."""
    return {
        "media_id": piece.media_id,
        "kind": piece.kind,
        # What this file can actually give. A clip — how long it runs, whole: a stretch out of the
        # middle of it, or two, can only be asked for by a director who knows there is a middle. A
        # photograph — as long as any piece may be held.
        "seconds_available": round(piece.duration or MAX_PIECE_SECONDS, 1),
        **known.get(piece.media_id, {}),
    }


def _thought(step):
    """Что режиссёр сказал об одном кадре — полями журнала (`journal.COLUMNS`).

    Рамка «куда смотрит» раскладывается по четырём колонкам: либо все четыре, либо ни одной, и
    другой `answers.planned` сюда не пропустит. Незаданного поля в реплике просто не будет.
    """
    return {
        "media_id": step["media_id"],
        "thought": step["thought"],
        "story_part": step["part"],
        "story_part_title": step["part_title"],
        **{f"focus_{side}": part for side, part in (step["focus"] or {}).items()},
    }


def _cut(piece, seconds=None, start=None):
    """The piece at the length the plan gave it and, out of a clip, from the second the plan named —
    within what the screen and the file allow.

    Both are finite numbers or None (`answers.number`). No length — a step that said nothing about
    it, or nothing that is a number — keeps the length отбор cut it at: the plan is about the order
    of the film first, and a missing number is not a reason to leave a piece out of it. No start
    keeps the clip cut around its best moment, as отбор cut it; a photograph has no stretch to
    start anywhere, it is shown whole.
    """
    if seconds is None:
        seconds = piece.seconds
    else:
        seconds = min(max(seconds, MIN_PIECE_SECONDS), MAX_PIECE_SECONDS)
    if start is not None and piece.kind == "video" and piece.duration:
        # The stretch the director named, pulled inside the file, becomes the moment the piece is
        # built around. Every later cut of it — held longer for the ordered length
        # (`selection.stretch`), to a beat (`rhythm.fit`), at ten minutes (`selection.cap`) — is
        # centred on `best_second`, and around отбор's moment it would slide off the stretch that
        # was asked for. The cover of a montage that opens on it is taken from there too.
        held = min(seconds, piece.duration)
        start = min(max(start, 0.0), piece.duration - held)
        piece = dataclasses.replace(piece, best_second=start + held / 2)
    # `rejected` goes with it: the piece is in the film, so nothing about it is turned down — the
    # same thing `selection.choose` does to a piece the organizer overruled it on.
    return dataclasses.replace(selection.cut_to(piece, seconds), rejected=None)
