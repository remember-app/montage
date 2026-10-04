"""The `media_understanding` job: what is in one photo or clip, and who — asked once and kept.

Понимание, the pipeline's second module (doc-03): a cloud model is shown one ready media item and
says what happens in it, who is in the frame, where and when it was shot, in what mood and with
what shot — and, at a clip, how the camera moves and which moments are worth having. The answer
lands in `media_understandings` under the item's id, as one sentence plus those fields, and is read
from there ever after — by the montage's analysis stage and by the director, and by the second
montage out of the same memory, which is why nobody pays to describe a frame twice (doc-03, «Второй
ролик из тех же медиа не описывает заново уже описанные кадры»).

Who is who across a memory is the model's own doing and not a face database of ours (decision-9):
every call carries the cast this memory has already named — «человек-1», «животное-1» — and the
model either recognises one of them or introduces the next by number. The cast grows one item at a
time, which is why this job is one item at a time too: the labels of everything described before it
are what make the label on this frame the same person's.

What is sent are the derivatives `media_preview` already made, never the bytes somebody uploaded: a
photo goes as its 2048px copy, a clip as a few stills cut out of its 720p preview.
ponytail: a clip whole would show the model movement — somebody blowing out candles rather than
somebody sitting at a table — and Gemini takes one (`ai.Gemini`). Stills instead because OpenAI
takes no video at all and Gemini charges a clip by its length (258 tokens a second, doc-11), so
three of them answer «что происходит и кто в кадре» for either provider at the price of three
photographs. Hand Gemini the clip itself when the director turns out to need the movement.

`media_items` belongs to the api's migrations; this module only reads rows by id.
"""

import logging
import pathlib
import re
import tempfile

from psycopg.types.json import Jsonb

from montage import ai, ffmpeg, storage
from montage.ai import answers

FRAMES_PER_CLIP = 3  # spread across the clip, never at its edges — see `_frames`

UUID = re.compile(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.IGNORECASE)

log = logging.getLogger("montage.understanding")


def media_understanding(conn, ref):
    """One ready media item described and the description kept — or nothing asked of anybody."""
    if conn.execute("SELECT 1 FROM media_understandings WHERE media_id = %s", (ref,)).fetchone():
        # The whole point of keeping them: a second montage out of the same memory, and the hourly
        # sweep, both come back here on frames that were described weeks ago.
        log.info("media item %s is described already, not asking again", ref)
        return

    brain = ai.AI(conn)
    if brain.provider is None:
        # Today's state — a model is nobody's until the human buys a key (decision-10). Checked
        # before anything is downloaded, and nothing is stored: the item is simply undescribed, and
        # the sweep describes it for real the hour after a key appears.
        log.info("no cloud model: media item %s stays undescribed", ref)
        return

    row = conn.execute(
        """SELECT memory_id, kind, preview_key FROM media_items
           WHERE id = %s AND status = 'ready' AND preview_key IS NOT NULL""",
        (ref,),
    ).fetchone()
    if row is None:
        # Deleted, still being previewed, failed — or an old photo whose large copy is not made yet
        # (`worker.sweep_photos_without_previews` is making it). Nothing to describe and nothing
        # wrong: the sweep comes back for it once there is something to look at.
        log.info("media item %s has nothing ready to describe, leaving it", ref)
        return
    memory_id, kind, preview_key = row

    with tempfile.TemporaryDirectory() as workdir:
        workdir = pathlib.Path(workdir)
        preview = workdir / "preview"
        storage.download(preview_key, preview)
        frames = _frames(preview, workdir) if kind == "video" else [_blob(preview)]
        described = brain.describe_media(frames, cast=_cast(conn, memory_id), ref=ref)

    description = answers.said(described.get("description"))
    if not description:
        # An answer without the one thing that was asked for. Raising retries the job like any other
        # failure; storing it would mark the frame described for good and hand the director a blank.
        raise RuntimeError(f"the model described nothing: {str(described)[:500]}")

    cast = _labels(described.get("cast"))
    conn.execute(
        """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels, details)
           VALUES (%s, %s, %s, %s, %s) ON CONFLICT (media_id) DO NOTHING""",
        (ref, str(memory_id), description, cast, Jsonb(answers.described(described))),
    )
    log.info("media item %s described, in frame: %s", ref, ", ".join(cast) or "nobody")


def frame_comments(conn, media_ids):
    """Что участники воспоминания написали к кадрам: `{media_id: [текст, …]}`.

    Единственное место, откуда они возьмутся: и разбор кадров, и режиссёр читают комментарии через
    неё, а не спрашивают базу каждый сам. Хранилище — `frame_comments` (миграция 190): пишут
    комментарии через api, воркер их только читает, и индекс `(media_id, created_at)` заведён
    ровно под этот отбор — комментарии кадра по времени написания, старые раньше новых.

    Кадра, которому ничего не написали, в ответе нет вовсе: пустой ответ — не ошибка и не повод
    придумывать, вызывающий код обязан быть готов к тому, что у любого кадра комментариев нет, и не
    должен этим догадываться — список пуст сам по себе.

    Отбор идёт по uuid, как его держит `frame_comments`, и мимо индекса на то же самое не уйти: на
    текстовом сравнении таблица читается вся. Поэтому id, которые uuid не есть, уходят в отсев
    здесь, а не падают на приведении в самом запросе.
    """
    said = {}
    for media_id, body in conn.execute(
        """SELECT media_id::text, body FROM frame_comments
           WHERE media_id = ANY(%s::uuid[])
           ORDER BY media_id, created_at""",
        ([one for one in media_ids if UUID.fullmatch(one)],),
    ):
        said.setdefault(media_id, []).append(body)
    return said


def describe_missing(conn, media_ids):
    """Описать те из этих кадров, которых ещё никто не описывал; вернуть взятые готовыми.

    Тот же `media_understanding`, что и у задачи из очереди, — поэтому кадр, описанный месяц назад
    или прошлым роликом из этого же воспоминания, в нейросеть второй раз не уходит: он и здесь, и
    в задаче начинается с одной и той же проверки. Возвращается множество тех, чьё описание взято
    из таблицы, — журнал помечает их «описан ранее», и по нему же считает, сколько описано
    сейчас (контракт api, «analysis»).
    """
    earlier = {
        (row[0],)
        for row in conn.execute(
            "SELECT media_id FROM media_understandings WHERE media_id = ANY(%s)", (list(media_ids),)
        )
    }
    for media_id in media_ids:
        media_understanding(conn, media_id)
    return {media_id for (media_id,) in earlier}


def _cast(conn, memory_id):
    """Everyone this memory has already named, for the model to recognise instead of renaming.

    Sorted rather than in the order they were met: the same memory then asks the same question about
    two different frames, which is one less thing to explain when two answers disagree.
    """
    return [
        label
        for (label,) in conn.execute(
            """SELECT DISTINCT unnest(cast_labels) AS label FROM media_understandings
               WHERE memory_id = %s ORDER BY label""",
            (str(memory_id),),
        ).fetchall()
    ]


def _labels(cast):
    """The labels out of the model's answer: strings out of a list, trimmed, in the order given,
    without repeats. Nothing else is a label — a string in place of the list would otherwise be kept
    on the frame for good, one letter per label."""
    named = (answers.said(name) for name in answers.listed(cast))
    return list(dict.fromkeys(name for name in named if name))


def _frames(preview, workdir):
    """A few stills out of one clip, spread evenly across it.

    None of them at the edges: the first frame of a phone video is regularly black (`media`), and
    the last one is as often the pocket the phone went into. A clip whose length ffmpeg will not
    report is described by its one frame rather than by three copies of it.
    """
    duration = ffmpeg.duration(preview) or 0.0
    count = FRAMES_PER_CLIP if duration else 1
    frames = []
    for index in range(count):
        at = duration * (index + 1) / (count + 1)
        frame = workdir / f"frame-{index}.jpg"
        ffmpeg.run("-ss", f"{at:.3f}", "-i", str(preview), "-frames:v", "1", str(frame))
        frames.append(_blob(frame))
    return frames


def _blob(path):
    """Everything sent to a model is a JPEG this worker made itself — a preview it encoded or a
    frame it cut out of one — so the type is known rather than guessed at from a file name."""
    return ai.Blob("image/jpeg", path.read_bytes())
