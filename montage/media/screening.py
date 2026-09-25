"""The `media_screening` job: what the отсев makes of one file, long before a video is ordered.

The organizer picks the material for a video out of a gallery that already says what the analysis
thinks of every frame in it — «автоматический анализ, который отмечает за организатора» (doc-03,
«Автоматический отсев брака как предложение»). That judgement therefore cannot wait for the order:
it is passed on each item as it becomes `ready` and kept on the row, in `screening_rejected` and
`screening_reason` (the api's migration `060.do.media-screening.sql`).

Nothing here looks at a frame itself. `selection.judge` is the one place that does, and it looks at
the file *alone* — which is exactly what makes it answerable this early. What comes back is a
**proposal**: the organizer overrules it in either direction before pressing «смонтировать», and
what actually goes into a video is still `selection.choose`'s answer at the order, honouring
«включить» and «исключить». Nothing in this module is read by the montage pipeline.

`duplicate` is deliberately not written here. A burst of shots two seconds apart is one moment, and
which of them stands for it is decided by comparing them with each other (`selection._dedup`, over
the whole memory at once) — a file is not a duplicate on its own, and a verdict that leaned on its
neighbours would flip every time somebody else uploaded a shot of the same second. The three reasons
this job can write — `unreadable`, `blur`, `exposure` — are the ones true of the file itself; the
organizer meets `duplicate` where it is decided, in the order.

`media_items` belongs to the api's migrations. This module only reads and writes rows by id.
"""

import logging
import pathlib
import tempfile

from montage import storage
from montage.video import selection

log = logging.getLogger("montage.screening")


def media_screening(conn, ref):
    """One item: the file out of R2, `judge`'s verdict onto the row."""
    row = conn.execute(
        "SELECT kind, original_key, preview_key FROM media_items WHERE id = %s", (ref,)
    ).fetchone()
    if row is None:
        # Deleted while the job waited — a member left and took their material with them. Nothing to
        # screen and nothing wrong: retrying would only fail three times.
        log.info("media item %s is gone, nothing to screen", ref)
        return
    kind, original_key, preview_key = row

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

    # `false` and not `null` for a frame that passed: null is «the analysis has not looked yet», and
    # the gallery has to be able to tell the two apart (the api's migration 060).
    conn.execute(
        """UPDATE media_items
           SET screening_rejected = %s, screening_reason = %s, updated_at = now()
           WHERE id = %s""",
        (judged.rejected is not None, judged.rejected, ref),
    )
    # The row's own `status` is not touched, unlike a preview that gives up: a file the analysis
    # throws out is still in the gallery, still downloadable, and still one «включить» away from the
    # video. Rubbish in somebody's eyes, maybe; broken, no.
    log.info("media item %s screened: %s", ref, judged.rejected or "kept")
