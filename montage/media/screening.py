import logging
import pathlib
import tempfile

from montage import storage
from montage.video import selection

log = logging.getLogger("montage.screening")


def media_analysis(conn, ref):
    screen_memory(conn, ref)


def media_screening(conn, ref):
    row = conn.execute(
        "SELECT id, kind, original_key, preview_key, memory_id FROM media_items WHERE id = %s",
        (ref,),
    ).fetchone()
    if row is None:
        log.info("media item %s is gone, nothing to screen", ref)
        return
    media_id, kind, original_key, preview_key, memory_id = row

    judged = _judged(media_id, kind, original_key, preview_key)
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (str(memory_id),))
        if not _written(conn, judged):
            return
        _duplicates(conn, memory_id)


def screen_memory(conn, memory_id, judged=None):
    missing = conn.execute(
        """SELECT item.id, item.kind, item.original_key, item.preview_key
           FROM media_items item
           WHERE item.memory_id = %s AND item.status = 'ready' AND NOT item.hidden
             AND (item.screening_rejected IS NULL
                  OR (item.screening_rejected = false AND NOT EXISTS (
                    SELECT 1 FROM media_frames seen WHERE seen.media_id = item.id::text)))
           ORDER BY coalesce(item.captured_at, item.created_at), item.id""",
        (str(memory_id),),
    ).fetchall()

    if not missing:
        _duplicates(conn, memory_id)
        return
    if judged is None:
        judged = [_judged(*row) for row in missing]
    else:
        waiting = {str(row[0]) for row in missing}
        judged = [piece for piece in judged if str(piece.media_id) in waiting]
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (str(memory_id),))
        for one in judged:
            _written(conn, one)
        _duplicates(conn, memory_id)


def _judged(media_id, kind, original_key, preview_key):
    media_id = str(media_id)
    with tempfile.TemporaryDirectory() as workdir:
        workdir = pathlib.Path(workdir)
        looked_at = None
        if kind == "photo" and preview_key:
            looked_at = storage.large_copy(preview_key, workdir / "large.jpg")
        if looked_at is None:
            looked_at = workdir / "original"
            storage.download(original_key, looked_at)
        return selection.judge(selection.Piece(media_id=media_id, kind=kind, path=looked_at))


def _written(conn, judged):
    written = conn.execute(
        """UPDATE media_items
           SET screening_rejected = %s, screening_reason = %s, updated_at = now()
           WHERE id = %s""",
        (judged.rejected is not None, judged.rejected, judged.media_id),
    ).rowcount
    if not written:
        log.info("media item %s went away mid-screening, nothing to write", judged.media_id)
        return False
    if judged.frame:
        conn.execute(
            """INSERT INTO media_frames (media_id, frame, sharpness) VALUES (%s, %s, %s)
               ON CONFLICT (media_id)
               DO UPDATE SET frame = excluded.frame, sharpness = excluded.sharpness""",
            (judged.media_id, judged.frame, judged.sharpness),
        )
    log.info("media item %s screened: %s", judged.media_id, judged.rejected or "kept")
    return True


def _duplicates(conn, memory_id):
    pieces = [
        selection.Piece(
            media_id=str(media_id), captured_at=captured_at, sharpness=sharpness, frame=frame
        )
        for media_id, captured_at, sharpness, frame in conn.execute(
            """SELECT item.id, item.captured_at, seen.sharpness, seen.frame
               FROM media_items item JOIN media_frames seen ON seen.media_id = item.id::text
               WHERE item.memory_id = %s AND item.status = 'ready' AND NOT item.hidden
                 AND (item.screening_rejected = false OR item.screening_reason = %s)
               ORDER BY coalesce(item.captured_at, item.created_at), item.id""",
            (memory_id, selection.DUPLICATE),
        )
    ]
    retakes = [piece.media_id for piece in selection._dedup(pieces)[1]]
    changed = conn.execute(
        """UPDATE media_items
           SET screening_rejected = (id = ANY(%(retakes)s::uuid[])),
               screening_reason = CASE WHEN id = ANY(%(retakes)s::uuid[]) THEN %(reason)s END,
               updated_at = now()
           WHERE id = ANY(%(material)s::uuid[])
             AND screening_rejected IS DISTINCT FROM (id = ANY(%(retakes)s::uuid[]))""",
        {
            "retakes": retakes,
            "material": [piece.media_id for piece in pieces],
            "reason": selection.DUPLICATE,
        },
    ).rowcount
    log.info(
        "memory %s: %s of %s kept frame(s) are retakes, %s verdict(s) changed",
        memory_id,
        len(retakes),
        len(pieces),
        changed,
    )
