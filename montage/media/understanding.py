import logging
import pathlib
import re
import tempfile

from psycopg.types.json import Jsonb

from montage import ai, ffmpeg, storage
from montage.ai import answers

FRAMES_PER_CLIP = 3

UUID = re.compile(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.IGNORECASE)

log = logging.getLogger("montage.understanding")


def media_understanding(conn, ref):
    if conn.execute("SELECT 1 FROM media_understandings WHERE media_id = %s", (ref,)).fetchone():
        log.info("media item %s is described already, not asking again", ref)
        return

    brain = ai.AI(conn)
    if brain.provider is None:
        log.info("no cloud model: media item %s stays undescribed", ref)
        return

    row = conn.execute(
        """SELECT memory_id, kind, preview_key FROM media_items
           WHERE id = %s AND status = 'ready' AND preview_key IS NOT NULL""",
        (ref,),
    ).fetchone()
    if row is None:
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
        raise RuntimeError(f"the model described nothing: {str(described)[:500]}")

    cast = _labels(described.get("cast"))
    conn.execute(
        """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels, details)
           VALUES (%s, %s, %s, %s, %s) ON CONFLICT (media_id) DO NOTHING""",
        (ref, str(memory_id), description, cast, Jsonb(answers.described(described))),
    )
    log.info("media item %s described, in frame: %s", ref, ", ".join(cast) or "nobody")


def frame_comments(conn, media_ids):
    said = {}
    for media_id, body in conn.execute(
        """SELECT media_id::text, body FROM frame_comments
           WHERE media_id = ANY(%s::uuid[])
           ORDER BY media_id, created_at""",
        ([one for one in media_ids if UUID.fullmatch(one)],),
    ):
        said.setdefault(media_id, []).append(body)
    return said


def describe_missing(conn, media_ids, failed=None):
    earlier = {
        (row[0],)
        for row in conn.execute(
            "SELECT media_id FROM media_understandings WHERE media_id = ANY(%s)", (list(media_ids),)
        )
    }
    for media_id in media_ids:
        try:
            media_understanding(conn, media_id)
        except Exception as exc:
            log.warning("media item %s is left undescribed: %s", media_id, exc)
            if failed is not None:
                failed.append(media_id)
    return {media_id for (media_id,) in earlier}


def _cast(conn, memory_id):
    return [
        label
        for (label,) in conn.execute(
            """SELECT DISTINCT unnest(cast_labels) AS label FROM media_understandings
               WHERE memory_id = %s ORDER BY label""",
            (str(memory_id),),
        ).fetchall()
    ]


def _labels(cast):
    named = (answers.said(name) for name in answers.listed(cast))
    return list(dict.fromkeys(name for name in named if name))


def _frames(preview, workdir):
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
    return ai.Blob("image/jpeg", path.read_bytes())
