"""Housekeeping the queue itself would never ask for: what nobody queued, and what nobody finished.

Every sweep runs off the worker's own loop, at start-up and then once an hour. Two kinds live here
and they answer to the same rule — do the least that brings the tables back to what they should
say, and never pile work on something already on its way:

    the catch-ups — ready material with no preview, or no size and camera. Everything taken in
    before the worker learned to do a thing is like this, and nothing else will ever come for it:
    the api queues a job at upload and the upload is long over. Only what is read without a model
    and without a decision of a person's: the отсев and everything a cloud model answers are asked
    for by «Analyze» or by an order (`docs/jobs/analysis.md`), never by the hour.

    the uploads nobody finished — a multipart started a day ago and abandoned, which costs the
    bucket until somebody aborts it.

    the retakes counted again — what an organizer does to a frame after it was screened, which
    moves the series it stands in and never queues a job (`sweep_media_duplicates`).
"""

import logging

from montage import storage
from montage.media import screening

STALE_UPLOAD_HOURS = 24  # a day in `pending_upload` means the phone is not coming back
CATCH_UP_AFTER_HOURS = 24  # how long a catch-up that failed is left alone before it is tried again

log = logging.getLogger("montage.sweeps")


def _catch_up(conn, type_, missing):
    """Queue `type_` for every ready item that `missing` picks out — and only where nothing is
    already doing it.

    `missing` is the condition over `media_items item` saying what this item has not got; it is the
    one thing that differs between the catch-ups below, and it is a constant of this module, never
    anything a request carries. Everything around it is the same for all of them, which is why it
    is written once:

    an item already `queued` or `running` is left to finish, so an hourly sweep does not pile jobs
    on one that is simply still working — and one whose catch-up has just been given up on is left
    alone for a day. Three attempts is the whole retry policy (`run_once`), and a sweep that queued
    a fresh job the next hour would make it endless: an original that is not in the bucket at all
    would be downloaded and encoded three times an hour for as long as the worker runs, and a frame
    the model chokes on would be *paid for* three times an hour. A day later it is tried once more,
    because what failed may as easily have been an hour of R2 being unreachable or a day's limit on
    the account, and one such hour must not cost every old item its large copy — or its description
    — for good. Returns how many were queued.
    """
    return conn.execute(
        f"""INSERT INTO jobs (type, ref)
            SELECT %s, item.id::text FROM media_items item
            WHERE item.status = 'ready' AND {missing}
              AND NOT EXISTS (
                SELECT 1 FROM jobs job
                WHERE job.type = %s AND job.ref = item.id::text
                  AND (job.status IN ('queued', 'running')
                       OR (job.status = 'failed'
                           AND job.updated_at > now() - make_interval(hours => %s))))""",
        (type_, type_, CATCH_UP_AFTER_HOURS),
    ).rowcount


def sweep_photos_without_previews(conn):
    """Photos that are `ready` but carry no large copy, given the job that makes one.

    Every photo taken in before the worker learned to make one is like this, and nothing else will
    ever come for them: the api queues `media_preview` at upload and the upload is long over. The
    original is still in the bucket, so this costs the member nothing — no re-upload, no prompt.
    """
    queued = _catch_up(conn, "media_preview", "item.kind = 'photo' AND item.preview_key IS NULL")
    if queued:
        log.info("queued a large copy for %s photo(s) that had none", queued)
    return queued


def sweep_media_without_frame_info(conn):
    """Ready material nobody has measured or read the camera off yet, given the job that does both.

    Everything taken in before the worker learned to read them is like this: `media_preview` queues
    the job as an item goes `ready`, and for old material that moment is long past. The original is
    still in the bucket, so this costs the member nothing — no re-upload, no prompt.

    The size is what says «this file has been read», and the camera could not: a frame whose camera
    wrote nothing down keeps those columns null — an ordinary answer, indistinguishable from an
    unread file — while a picture that is in the gallery at all has a width, `media_preview` having
    made its thumbnail out of one. So the original is fetched once and the item is left alone
    after, and a catch-up that failed all three times gets its one more go a day later, like every
    other catch-up here.

    A file the size cannot be had of (the job ends `done`, the width stays null) would be the one
    that never stops: `_catch_up` lets a `done` job through, so it would be fetched and probed every
    hour for as long as the worker runs. A read that finished within the day is left alone like a
    failed one, and the same file costs one fetch a day.
    """
    queued = _catch_up(
        conn,
        "media_frame_info",
        f"""item.width IS NULL AND NOT EXISTS (
        SELECT 1 FROM jobs finished
        WHERE finished.type = 'media_frame_info' AND finished.ref = item.id::text
          AND finished.status = 'done'
          AND finished.updated_at > now() - make_interval(hours => {CATCH_UP_AFTER_HOURS}))""",
    )
    if queued:
        log.info("queued a size, a camera and a place for %s ready item(s) that had none", queued)
    return queued


def sweep_media_duplicates(conn):
    """Every memory's series counted again over its material as it stands now.

    A `duplicate` is not a verdict about a frame on its own — it is a verdict about a frame *among
    its neighbours* (`screening._duplicates`), and a neighbour can change without a single new file
    arriving: the organizer hides a frame or deletes one (`frame.ts`, `gallery.ts`), or corrects the
    date it was shot on, and any of the three moves the series every other frame of that memory
    stands in. None of them queues a job — they are the api's, and it queues `media_preview` at
    upload, `media_analysis` on «Analyze» and `montage` at the order — so without this sweep a
    «Дубль» would stand until the
    next upload into that same memory, and one that never comes would stand for good.

    Counted from scratch by the same `_duplicates` a screened file runs, over the same material, so
    the answer is the one the order would give: hidden is not material, and a frame that is no
    longer a retake is simply kept again. Written only where it changed, which for a memory nobody
    touched is nowhere.

    ponytail: every memory with material, on the hour. What would narrow it is a record of what the
    api changed, and there is none — a deleted frame leaves no row behind to notice — so the only
    honest cheapness here is to count what there is and let the write be a no-op. Re-count on the
    api's own events when it starts saying what it changed.
    """
    memories = conn.execute(
        """SELECT DISTINCT item.memory_id FROM media_items item
           JOIN media_frames seen ON seen.media_id = item.id::text
           WHERE item.status = 'ready'"""
    ).fetchall()
    for (memory_id,) in memories:
        screening._duplicates(conn, memory_id)
    if memories:
        log.info("counted the retakes again over %s memories", len(memories))
    return len(memories)


def sweep_stale_uploads(conn):
    """Uploads that were started and never finished, once an hour off the worker's own loop.

    An abandoned multipart is not free — the parts already sent sit in the bucket until someone
    aborts it. Returns how many rows were swept.
    """
    stale = conn.execute(
        """SELECT id, original_key, upload_id FROM media_items
           WHERE status = 'pending_upload' AND created_at < now() - make_interval(hours => %s)""",
        (STALE_UPLOAD_HOURS,),
    ).fetchall()
    if not stale:
        return 0

    # Built before anything is claimed: storage that cannot be built for want of a setting must not
    # cost the rows, which are the only record of what is left in the bucket.
    storage.client()
    swept = 0
    for item_id, original_key, upload_id in stale:
        # The row is claimed first, and only if it is still the row that was found. A phone can
        # finish a day-old upload in the seconds between the SELECT above and this line — the api
        # then moves it to `processing` and counts its bytes — and freeing R2 for that one would
        # delete a live original and leave the memory's counters permanently over.
        claimed = conn.execute(
            "DELETE FROM media_items WHERE id = %s AND status = 'pending_upload' RETURNING id",
            (item_id,),
        ).fetchone()
        if claimed is None:
            log.info("upload %s finished while the sweep ran, leaving it alone", item_id)
            continue
        try:
            # The parts, and whatever landed under the key anyway, for the client that completed
            # its upload in the bucket and never told the api.
            storage.free(original_key, upload_id)
        except Exception as exc:
            # The row is already gone, so nothing will ever name these again and the next hour will
            # not find them. The log is the only record left — hence the upload id in it.
            log.warning(
                "swept %s but could not free it in the bucket (upload %s), left behind: %s",
                original_key,
                upload_id,
                exc,
            )
        swept += 1

    log.info("swept %s abandoned upload(s) of %s found", swept, len(stale))
    return swept


SWEEPS = (
    sweep_stale_uploads,
    sweep_photos_without_previews,
    sweep_media_without_frame_info,
    sweep_media_duplicates,
)
