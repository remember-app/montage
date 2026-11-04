import logging
import pathlib
import tempfile

from montage import ffmpeg, photo, storage

THUMBNAIL_WIDTH = 640
PREVIEW_HEIGHT = 720
PREVIEW_LONG_SIDE = 2048

THUMBNAIL_SCALE = f"scale='min({THUMBNAIL_WIDTH},iw)':-2"
PREVIEW_SCALE = f"scale=-2:'min({PREVIEW_HEIGHT},ih)'"

PHOTO_PREVIEW_SCALE = (
    f"scale=w='min({PREVIEW_LONG_SIDE},iw)':h='min({PREVIEW_LONG_SIDE},ih)'"
    ":force_original_aspect_ratio=decrease:force_divisible_by=2"
)

log = logging.getLogger("montage.media.preview")


def media_preview(conn, ref):
    row = conn.execute(
        "SELECT kind, original_key FROM media_items WHERE id = %s", (ref,)
    ).fetchone()
    if row is None:
        log.info("media item %s is gone, nothing to preview", ref)
        return
    kind, original_key = row

    thumbnail_key, preview_key, duration = f"thumbnails/{ref}.jpg", None, None
    with tempfile.TemporaryDirectory() as workdir:
        workdir = pathlib.Path(workdir)
        original, thumbnail = workdir / "original", workdir / "thumbnail.jpg"
        storage.download(original_key, original)
        ffmpeg.refuse_playlists(original)

        preview = workdir / ("preview.mp4" if kind == "video" else "preview.jpg")
        if kind == "video":
            duration = ffmpeg.duration(original)
            at = min(1.0, duration / 2) if duration else 0
            ffmpeg.run(
                "-ss",
                str(at),
                "-i",
                str(original),
                "-frames:v",
                "1",
                "-vf",
                THUMBNAIL_SCALE,
                str(thumbnail),
            )
            ffmpeg.run(
                "-i",
                str(original),
                "-vf",
                PREVIEW_SCALE,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(preview),
            )
            preview_key = f"previews/{ref}.mp4"
            storage.upload(preview, preview_key, "video/mp4")
        else:
            upright = workdir / "upright.jpg"
            seen = upright if photo.stand_up(original, upright) else original
            ffmpeg.run("-i", str(seen), "-frames:v", "1", "-vf", THUMBNAIL_SCALE, str(thumbnail))
            ffmpeg.run(
                "-i",
                str(seen),
                "-frames:v",
                "1",
                "-vf",
                PHOTO_PREVIEW_SCALE,
                "-q:v",
                "3",
                str(preview),
            )
            preview_key = f"previews/{ref}.jpg"
            storage.upload(preview, preview_key, "image/jpeg")

        storage.upload(thumbnail, thumbnail_key, "image/jpeg")

    landed = conn.execute(
        """UPDATE media_items
           SET status = 'ready', thumbnail_key = %s, preview_key = %s,
               duration_seconds = coalesce(%s, duration_seconds), updated_at = now()
           WHERE id = %s
           RETURNING id""",
        (thumbnail_key, preview_key, duration, ref),
    ).fetchone()
    if landed is None:
        log.info("media item %s went away mid-preview, taking its derivatives back", ref)
        for key in (thumbnail_key, preview_key):
            if key:
                storage.delete(key)
        return
    log.info("media item %s (%s) is ready", ref, kind)
    follow_ups = ["media_frame_info"]
    conn.execute(
        """INSERT INTO jobs (type, ref)
           SELECT follow_up, %(ref)s FROM unnest(%(types)s::text[]) follow_up
           WHERE NOT EXISTS (
             SELECT 1 FROM jobs job
             WHERE job.type = follow_up AND job.ref = %(ref)s
               AND job.status IN ('queued', 'running'))""",
        {"ref": ref, "types": follow_ups},
    )


def give_up(conn, ref):
    conn.execute(
        "UPDATE media_items SET status = 'failed', updated_at = now() "
        "WHERE id = %s AND status <> 'ready'",
        (ref,),
    )
