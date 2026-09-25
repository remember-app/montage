"""Housekeeping the queue itself would never ask for: what nobody queued, and what nobody finished.

Every sweep runs off the worker's own loop, at start-up and then once an hour. Two kinds live here
and they answer to the same rule — do the least that brings the tables back to what they should
say, and never pile work on something already on its way:

    the catch-ups — ready material with no preview, no size, no screening proposal, nothing the
    cloud model was ever asked about. Everything taken in before the worker learned to do a thing
    is like this, and nothing else will ever come for it: the api queues a job at upload and the
    upload is long over.

    the uploads nobody finished — a multipart started a day ago and abandoned, which costs the
    bucket until somebody aborts it.
"""

import logging

from montage import ai, storage

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


def sweep_media_without_screening(conn):
    """Ready material the отсев has not looked at yet, given the job that looks.

    Everything taken in before the worker learned to screen is like this: the proposal is queued the
    moment an item goes `ready` (`preview.media_preview`), and for old material that moment is long
    past. Until the job has run both columns are `null` — «не посчитано», which the gallery shows as
    no opinion at all rather than as a frame the analysis kept.
    """
    queued = _catch_up(conn, "media_screening", "item.screening_rejected IS NULL")
    if queued:
        log.info("queued a screening proposal for %s ready item(s) that had none", queued)
    return queued


def sweep_media_without_ai_answers(conn):
    """Ready media the cloud model was never asked about: frames with no description, clips with no
    transcript.

    Both jobs are queued at the end of `preview.media_preview` and both are lost the same way. One:
    the item became ready while there was no key at all — the job then stored nothing and left the
    item as it was. Two, and this is the one that bites: the model refused. An account over its free
    limit answers «429» to every call, three attempts go in a minute and the job is `failed` for
    good — and nothing ever comes back for that frame, because ordering a video does not describe
    anything, it only reads what is already in the tables (doc-03). A memory then holds a hundred
    failed descriptions and an AI montage that will never turn on, however often it is re-ordered —
    which is a thing that has happened here, on a free tier. This sweep is what comes back.

    Hence the first line — with no model configured there is simply nothing to do, and without it
    the whole library would be queued every hour for every job of it to answer «nobody to ask». The
    two are asked in one sweep so that a worker without a key says so once an hour and not twice.

    ponytail: the whole library at once, in one insert each. A cap per sweep is the knob to add when
    a real key makes a first sweep over a year of memories a bill rather than a queue.
    """
    if ai.from_env() is None:
        return 0
    # A description is made out of the copies `media_preview` left, never the original — a frame
    # whose large copy is not made yet is `sweep_photos_without_previews`'s first.
    described = _catch_up(
        conn,
        "media_understanding",
        """item.preview_key IS NOT NULL AND NOT EXISTS (
             SELECT 1 FROM media_understandings seen WHERE seen.media_id = item.id::text)""",
    )
    if described:
        log.info("queued a description for %s media item(s) that had none", described)
    # Speech is a clip's alone, and an empty row is an answer (`speech.py`): a silent clip has one
    # and is not asked about again.
    transcribed = _catch_up(
        conn,
        "media_speech",
        """item.kind = 'video' AND NOT EXISTS (
             SELECT 1 FROM media_speech said WHERE said.media_id = item.id::text)""",
    )
    if transcribed:
        log.info("queued a transcript for %s clip(s) that had none", transcribed)
    return described + transcribed


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
    sweep_media_without_screening,
    sweep_media_without_ai_answers,
)
