"""The `media_preview` job: what the gallery actually shows, made out of the file that was uploaded.

The api writes a `media_items` row, takes the bytes into R2 and queues the job; from `processing`
onward the row is this worker's (see the api's migration `004.do.media.sql`). Here the original is
pulled back out, ffmpeg makes a thumbnail for the grid and, for both kinds, a lighter full-screen
copy (2048px JPEG for a photo, 720p mp4 for a video), and the row is left `ready` with the
derivative keys on it — and with the jobs that come after it queued.

`media_items` belongs to the api's migrations. This module only reads and writes rows by id.
"""

import logging
import pathlib
import tempfile

from montage import ffmpeg, photo, storage

THUMBNAIL_WIDTH = 640  # the gallery grid
PREVIEW_HEIGHT = 720  # «версия 720p H.264» (architecture.md)
PREVIEW_LONG_SIDE = 2048  # a photo opened full screen on a phone, retina included

# `min(…, iw)` rather than a flat width: a small original is left at its own size instead of being
# blown up into a blurrier, heavier file than the thing it previews. `-2` keeps the aspect ratio and
# rounds to an even number, which H.264 and yuv420p JPEG both insist on.
THUMBNAIL_SCALE = f"scale='min({THUMBNAIL_WIDTH},iw)':-2"
PREVIEW_SCALE = f"scale=-2:'min({PREVIEW_HEIGHT},ih)'"

# A photo is landscape or portrait, so the cap is on the long side rather than on the width: a box
# of the original's own size clipped to 2048, and `decrease` fits inside it without ever growing it.
PHOTO_PREVIEW_SCALE = (
    f"scale=w='min({PREVIEW_LONG_SIDE},iw)':h='min({PREVIEW_LONG_SIDE},ih)'"
    ":force_original_aspect_ratio=decrease:force_divisible_by=2"
)

log = logging.getLogger("montage.media.preview")


def media_preview(conn, ref):
    """One media item: the original out of R2, its derivatives back in, the row to `ready`."""
    row = conn.execute(
        "SELECT kind, original_key FROM media_items WHERE id = %s", (ref,)
    ).fetchone()
    if row is None:
        # Deleted while the job waited — a member left and took their material with them. Nothing
        # to preview and nothing wrong: retrying would only fail three times.
        log.info("media item %s is gone, nothing to preview", ref)
        return
    kind, original_key = row

    s3 = storage.client()
    thumbnail_key, preview_key, duration = f"thumbnails/{ref}.jpg", None, None
    with tempfile.TemporaryDirectory() as workdir:
        workdir = pathlib.Path(workdir)
        original, thumbnail = workdir / "original", workdir / "thumbnail.jpg"
        s3.download_file(storage.bucket(), original_key, str(original))
        # Before ffmpeg is pointed at it for anything else: these bytes came off somebody's phone
        # and nothing between there and here has looked inside them.
        ffmpeg.refuse_playlists(original)

        preview = workdir / ("preview.mp4" if kind == "video" else "preview.jpg")
        if kind == "video":
            duration = ffmpeg.duration(original)
            # A frame a little way in: the first one of a phone video is regularly black, and half
            # way is the fallback for a clip shorter than a second.
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
                # The gallery plays this progressively (architecture.md): the index has to sit at
                # the front of the file or the player has to fetch all of it before the first frame.
                "-movflags",
                "+faststart",
                str(preview),
            )
            preview_key = f"previews/{ref}.mp4"
            storage.upload(s3, preview, preview_key, "video/mp4")
        else:
            # A HEIC goes to ffmpeg already stood up: it does not apply the file's own turn
            # (`photo.stand_up`), and JPEG's EXIF orientation it does.
            upright = workdir / "upright.jpg"
            seen = upright if photo.stand_up(original, upright) else original
            ffmpeg.run("-i", str(seen), "-frames:v", "1", "-vf", THUMBNAIL_SCALE, str(thumbnail))
            # The full-screen copy. The original stays where it is — it is what the montage is cut
            # from — but a phone must not have to pull forty megapixels down to look at one frame.
            # `-q:v 3` because the whole point of this file is that it is sharp.
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
            storage.upload(s3, preview, preview_key, "image/jpeg")

        storage.upload(s3, thumbnail, thumbnail_key, "image/jpeg")

    # The measured length replaces the one the phone declared at intent, which was only ever a
    # claim. `width`/`height` are not written here but by `media_frame_info` below, which is queued
    # at the end of this handler: they are read off the original like the camera and the place, and
    # one owner for the lot means one answer and one catch-up.
    landed = conn.execute(
        """UPDATE media_items
           SET status = 'ready', thumbnail_key = %s, preview_key = %s,
               duration_seconds = coalesce(%s, duration_seconds), updated_at = now()
           WHERE id = %s
           RETURNING id""",
        (thumbnail_key, preview_key, duration, ref),
    ).fetchone()
    if landed is None:
        # Deleted while this job was running — the member walked out, or somebody deleted the item.
        # The api drops the keys the *row* carried, and these two were not on it yet, so nothing
        # else will ever come for them.
        log.info("media item %s went away mid-preview, taking its derivatives back", ref)
        for key in (thumbnail_key, preview_key):
            if key:
                s3.delete_object(Bucket=storage.bucket(), Key=key)
        return
    log.info("media item %s (%s) is ready", ref, kind)
    # What comes after the copies, queued by hand rather than through `worker.enqueue` — `worker`
    # imports this module, and the import cannot go both ways:
    #   the screening proposal, every kind's (`screening.py`): the organizer meets it in the gallery
    #   long before any video is ordered;
    #   speech, a clip's alone (`speech.py`) — a cloud model listening to one takes minutes, which
    #   the gallery does not wait on;
    #   what the camera wrote into the original — its model, the optics, the place — for the sheet
    #   «Сведения о кадре» (`frame_info.py`);
    #   and what is in the frame (`understanding.py`): it is describable the moment the copies
    #   exist, and the director wants it described long before anybody orders a video.
    # Only what is not already waiting: this job runs again whenever its worker died between here
    # and marking it done (`worker.reap`), and must not queue each of them twice.
    follow_ups = ["media_screening", "media_speech", "media_frame_info", "media_understanding"]
    if kind != "video":
        follow_ups.remove("media_speech")
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
    """Out of retries. The item is failed too, so the gallery stops waiting on it and shows a
    placeholder with «повторить» (architecture.md, «Обработка ошибок»).

    Unless it is already `ready`: the catch-up sweep runs this job again on photos that have been in
    the gallery for weeks, and failing to make one more derivative must not take a picture that
    works out of it. Those keep the thumbnail they have and stay openable.
    """
    conn.execute(
        "UPDATE media_items SET status = 'failed', updated_at = now() "
        "WHERE id = %s AND status <> 'ready'",
        (ref,),
    )
