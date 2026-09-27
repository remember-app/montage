"""The `media_screening` job: what the отсев makes of one file, long before a video is ordered.

The organizer picks the material for a video out of a gallery that already says what the analysis
thinks of every frame in it — «автоматический анализ, который отмечает за организатора» (doc-03,
«Автоматический отсев брака как предложение»). That judgement therefore cannot wait for the order:
it is passed on each item as it becomes `ready` and kept on the row, in `screening_rejected` and
`screening_reason` (the api's migration `060.do.media-screening.sql`).

Two questions are asked, one after the other. The first is of the file alone — `selection.judge`,
the one place that looks at a frame — and its answers are `unreadable`, `blur`, `exposure` or kept.
The second is of its neighbours: whether it is one more shot of a moment another file stands for,
`duplicate`. That is `selection._dedup`, the one the order runs, over the same material — ready,
not hidden, kept by the first question, read in the same order — so a frame marked here is the one
the montage folds into its series. A verdict that leans on neighbours changes when a neighbour
comes, so it is counted over the whole memory afresh every time a file is screened, and a frame a
new shot took the mark off is simply kept again.

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


def media_screening(conn, ref):
    """One item: the file out of R2, `judge`'s verdict onto the row — and its memory's retakes
    counted again with it among them."""
    row = conn.execute(
        "SELECT kind, original_key, preview_key, memory_id FROM media_items WHERE id = %s", (ref,)
    ).fetchone()
    if row is None:
        # Deleted while the job waited — a member left and took their material with them. Nothing to
        # screen and nothing wrong: retrying would only fail three times.
        log.info("media item %s is gone, nothing to screen", ref)
        return
    kind, original_key, preview_key, memory_id = row

    with tempfile.TemporaryDirectory() as workdir:
        workdir = pathlib.Path(workdir)
        # A photograph is judged on the large copy the preview made of it (2048px, `media_preview`)
        # and not on the forty megapixels it was uploaded at: the same picture, decoded in a tenth
        # of the time — and the same file the montage cuts the piece from, so the proposal and the
        # order are made on one and the same frame. No copy yet, or one the bucket will not give
        # back: the original, which is what this job has always looked at.
        #
        # A video stays on its original whatever copy exists beside it: its preview is a 720p
        # re-encode, which smooths away the very edges sharpness is measured in.
        looked_at = None
        if kind == "photo" and preview_key:
            looked_at = storage.large_copy(preview_key, workdir / "large.jpg")
        if looked_at is None:
            looked_at = workdir / "original"
            storage.download(original_key, looked_at)
        judged = selection.judge(selection.Piece(media_id=str(ref), kind=kind, path=looked_at))

    with conn.transaction():
        # One memory's screenings one after another, each over what the ones before it wrote: two
        # shots of one moment screened at once by two workers would otherwise each count the series
        # without the other, and whichever wrote last would leave its half-answer standing.
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (str(memory_id),))
        # `false` and not `null` for a frame that passed: null is «the analysis has not looked yet»,
        # and the gallery has to be able to tell the two apart (the api's migration 060).
        written = conn.execute(
            """UPDATE media_items
               SET screening_rejected = %s, screening_reason = %s, updated_at = now()
               WHERE id = %s""",
            (judged.rejected is not None, judged.rejected, ref),
        ).rowcount
        if not written:
            log.info("media item %s went away mid-screening, nothing to write", ref)
            return
        if judged.frame:
            conn.execute(
                """INSERT INTO media_frames (media_id, frame, sharpness) VALUES (%s, %s, %s)
                   ON CONFLICT (media_id)
                   DO UPDATE SET frame = excluded.frame, sharpness = excluded.sharpness""",
                (str(ref), judged.frame, judged.sharpness),
            )
        # The row's own `status` is not touched, unlike a preview that gives up: a file the analysis
        # throws out is still in the gallery, still downloadable, and still one «включить» away from
        # the video. Rubbish in somebody's eyes, maybe; broken, no.
        log.info("media item %s screened: %s", ref, judged.rejected or "kept")
        _duplicates(conn, memory_id)


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
