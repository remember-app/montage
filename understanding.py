"""The `media_understanding` job: what is in one photo or clip, and who — asked once and kept.

Понимание, the pipeline's second module (doc-03): a cloud model is shown one ready media item and
says in a sentence or two what happens in it and who is in the frame. The answer lands in
`media_understandings` under the item's id and is read from there ever after — by the director, and
by the second montage out of the same memory, which is why nobody pays to describe a frame twice
(doc-03, «Второй ролик из тех же медиа не описывает заново уже описанные кадры»).

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
import tempfile

import ai
import media

FRAMES_PER_CLIP = 3  # spread across the clip, never at its edges — see `_frames`

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
        media.storage().download_file(media.bucket(), preview_key, str(preview))
        frames = _frames(preview, workdir) if kind == "video" else [_blob(preview)]
        described = brain.describe_media(frames, cast=_cast(conn, memory_id), ref=ref)

    description = ai._said(described.get("description"))
    if not description:
        # An answer without the one thing that was asked for. Raising retries the job like any other
        # failure; storing it would mark the frame described for good and hand the director a blank.
        raise RuntimeError(f"the model described nothing: {str(described)[:500]}")

    cast = _labels(described.get("cast"))
    conn.execute(
        """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
           VALUES (%s, %s, %s, %s) ON CONFLICT (media_id) DO NOTHING""",
        (ref, str(memory_id), description, cast),
    )
    log.info("media item %s described, in frame: %s", ref, ", ".join(cast) or "nobody")


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
    named = (ai._said(name) for name in ai._listed(cast))
    return list(dict.fromkeys(name for name in named if name))


def _frames(preview, workdir):
    """A few stills out of one clip, spread evenly across it.

    None of them at the edges: the first frame of a phone video is regularly black (`media`), and
    the last one is as often the pocket the phone went into. A clip whose length ffmpeg will not
    report is described by its one frame rather than by three copies of it.
    """
    duration = media._duration(preview) or 0.0
    count = FRAMES_PER_CLIP if duration else 1
    frames = []
    for index in range(count):
        at = duration * (index + 1) / (count + 1)
        frame = workdir / f"frame-{index}.jpg"
        media._ffmpeg("-ss", f"{at:.3f}", "-i", str(preview), "-frames:v", "1", str(frame))
        frames.append(_blob(frame))
    return frames


def _blob(path):
    """Everything sent to a model is a JPEG this worker made itself — a preview it encoded or a
    frame it cut out of one — so the type is known rather than guessed at from a file name."""
    return ai.Blob("image/jpeg", path.read_bytes())
