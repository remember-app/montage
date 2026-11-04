import logging

from montage import storage
from montage.media import screening

STALE_UPLOAD_HOURS = 24
CATCH_UP_AFTER_HOURS = 24

log = logging.getLogger("montage.sweeps")


def _catch_up(conn, type_, missing):
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
    queued = _catch_up(conn, "media_preview", "item.kind = 'photo' AND item.preview_key IS NULL")
    if queued:
        log.info("queued a large copy for %s photo(s) that had none", queued)
    return queued


def sweep_media_without_frame_info(conn):
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
    stale = conn.execute(
        """SELECT id, original_key, upload_id FROM media_items
           WHERE status = 'pending_upload' AND created_at < now() - make_interval(hours => %s)""",
        (STALE_UPLOAD_HOURS,),
    ).fetchall()
    if not stale:
        return 0

    storage.client()
    swept = 0
    for item_id, original_key, upload_id in stale:
        claimed = conn.execute(
            "DELETE FROM media_items WHERE id = %s AND status = 'pending_upload' RETURNING id",
            (item_id,),
        ).fetchone()
        if claimed is None:
            log.info("upload %s finished while the sweep ran, leaving it alone", item_id)
            continue
        try:
            storage.free(original_key, upload_id)
        except Exception as exc:
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
