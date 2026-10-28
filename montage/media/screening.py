"""The отсев: what the analysis makes of a memory's material, asked for only when somebody asks.

Nothing here runs at upload. Two things start it, and both are a person's request: the organizer's
«Analyze» on the order screen (the api queues `media_analysis` with the memory's id in `ref`) and an
order itself, whose first stage fills in what is missing before anything is chosen (`montage.py`,
`analysis`). An order nobody analysed must still be assembled, so the montage gets the отсев it
needs itself.

Every verdict is local — `selection.judge` reads a file with ffmpeg and `selection._dedup` counts
the series — and no cloud model is asked. It is written onto the row in `screening_rejected` and
`screening_reason` (the api's migration `060.do.media-screening.sql`) and the card it is counted by
in `media_frames` (`schema.sql`).

What comes back is a **proposal**: the organizer overrules it in either direction before pressing
«смонтировать», and what actually goes into a video is still `selection.choose`'s answer at the
order, honouring «включить» and «исключить». Nothing in this module is read by the montage pipeline.

`media_items` belongs to the api's migrations. This module only reads and writes rows by id.
"""

import logging
import pathlib
import tempfile

from montage import storage
from montage.video import selection

log = logging.getLogger("montage.screening")


def media_analysis(conn, ref):
    """The `media_analysis` job: one memory's ready, visible material screened in one pass.

    `ref` is the memory's id — the api puts nothing smaller in the queue, because a series is a
    verdict about a frame *among its neighbours* and half a memory cannot answer for it.
    """
    screen_memory(conn, ref)


def media_screening(conn, ref):
    """One item: the file out of R2, `judge`'s verdict onto the row — and its memory's retakes
    counted again with it among them.

    Kept as a job of its own for a single-frame request; the analysis job below is what screens a
    whole memory without paying for the same recount per file.
    """
    row = conn.execute(
        "SELECT id, kind, original_key, preview_key, memory_id FROM media_items WHERE id = %s",
        (ref,),
    ).fetchone()
    if row is None:
        # Deleted while the job waited — a member left and took their material with them. Nothing to
        # screen and nothing wrong: retrying would only fail three times.
        log.info("media item %s is gone, nothing to screen", ref)
        return
    media_id, kind, original_key, preview_key, memory_id = row

    judged = _judged(media_id, kind, original_key, preview_key)
    with conn.transaction():
        # One memory's screenings one after another, each over what the ones before it wrote: two
        # shots of one moment screened at once by two workers would otherwise each count the series
        # without the other, and whichever wrote last would leave its half-answer standing.
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (str(memory_id),))
        if not _written(conn, judged):
            return
        _duplicates(conn, memory_id)


def screen_memory(conn, memory_id, judged=None):
    """Judge every ready, visible item of a memory that has no verdict yet, then count its series
    once — not once per file.

    The material is the order's: `ready`, not hidden, in the gallery's order. A file already judged
    is not fetched or decoded again; a kept file always leaves its frame behind, so one that has
    both is done, and the legacy case of a verdict without a frame is the exception the second half
    of the condition catches. A frame the first question turned down takes no part in a series, so
    it is done with its verdict alone.

    `judged` is the same judgement when the caller already has it: an order's preparation judges
    every file before this stage (`montage.py`), and fetching the whole memory again to judge it
    twice would be a second download and a second decode of every video. Only the items still
    missing a verdict are written.

    The series are recounted over the whole memory from what is stored (`_duplicates`), whether or
    not a file was judged now: a neighbour may have moved while nothing was left to read.
    """
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
        # The files are read outside the transaction: a download is slow, and holding a lock over
        # the whole memory while one ffmpeg runs would block the other worker that is screening it.
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
    """One file looked at, on the copy the piece would be cut from where there is one.

    A photograph is judged on the large copy the preview made of it (2048px, `media_preview`) and
    not on the forty megapixels it was uploaded at: the same picture, decoded in a tenth of the
    time — and the same file the montage cuts the piece from, so the proposal and the order are made
    on one and the same frame. No copy yet, or one the bucket will not give back: the original.

    A video stays on its original whatever copy exists beside it: its preview is a 720p re-encode,
    which smooths away the very edges sharpness is measured in.
    """
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
    """The verdict and the frame onto the item's row — false when the item went away mid-screen."""
    # `false` and not `null` for a frame that passed: null is «the analysis has not looked yet»,
    # and the gallery has to be able to tell the two apart (the api's migration 060).
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
    # The row's own `status` is not touched, unlike a preview that gives up: a file the analysis
    # throws out is still in the gallery, still downloadable, and still one «включить» away from
    # the video. Rubbish in somebody's eyes, maybe; broken, no.
    log.info("media item %s screened: %s", judged.media_id, judged.rejected or "kept")
    return True


def _duplicates(conn, memory_id):
    """Every retake in the memory marked `duplicate`, and every frame that is no longer one kept.

    Counted from scratch and never patched: a series is built in time order from its first shot
    (`selection._dedup`), so one new file can move a frame into a series or out of one, and the only
    answer that holds is the one counted over the memory as it is now.

    The material is the order's, taken the way the order takes it (`montage.py`) — ready, not
    hidden, in the gallery's order, so a tie between two shots falls the same way — less what the
    file alone was turned down for, which `selection.choose` puts aside before its dedup too. A
    frame taken off the list has nothing to go back to but «kept»: only kept frames join a series
    here. Nor is anybody's word undone: the organizer's «включить» lives on their order
    (`include_media_ids`), never on this row.
    """
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
