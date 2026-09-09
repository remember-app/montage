"""The `media_preview` job, and the hourly sweep of uploads that were never finished.

The api writes a `media_items` row, takes the bytes into R2 and queues the job; from `processing`
onward the row is this worker's (see the api's migration `004.do.media.sql`). Here the original is
pulled back out, `ffmpeg` makes what the gallery actually shows — a thumbnail for the grid and, for
both kinds, a lighter full-screen copy (2048px JPEG for a photo, 720p mp4 for a video) — and the row
is left `ready` with the derivative keys on it.

`media_items` belongs to the api's migrations. This module only reads and writes rows by id.
"""

import logging
import os
import pathlib
import subprocess
import tempfile

import boto3
from botocore.config import Config

STALE_UPLOAD_HOURS = 24  # a day in `pending_upload` means the phone is not coming back

# Containers that are not media but a list of *other* files to go and fetch. ffmpeg picks its
# demuxer by probing the bytes, and the api never looks inside an upload — `kind` is the phone's
# word for it — so a «photo» whose bytes are an HLS playlist would have ffmpeg open whatever paths
# the playlist names and encode them into a preview the uploader can then open from the gallery.
# That is a read of this worker's own disk, which is where DATABASE_URL and the R2 keys live.
#
# A denylist and not an allowlist on purpose: format names vary by build and by camera, and being
# wrong here must mean «one exotic container slips through», never «every photo is refused».
PLAYLIST_FORMATS = {"hls", "applehttp", "concat", "ffconcat", "dash", "sdp", "rtsp", "rtp"}
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

log = logging.getLogger("montage.media")


# --- R2 ---------------------------------------------------------------------
# The same four variable names the api reads, so one `.env` serves both.


def storage():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"].rstrip("/"),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        # R2 ignores the region and wants the literal `auto`; MinIO signs with `us-east-1` unless
        # told otherwise. A knob because those two disagree — same as the api.
        region_name=os.environ.get("R2_REGION", "auto"),
        # `<endpoint>/<bucket>/<key>`, which is how the api addresses it too. The alternative puts
        # the bucket in the hostname, and `remember.localhost` resolves nowhere.
        config=Config(s3={"addressing_style": "path"}),
    )


def bucket():
    return os.environ["R2_BUCKET"]


# --- ffmpeg -----------------------------------------------------------------


def _run(*args):
    """A tool, its output, and its stderr in the exception when it fails — that text is what ends
    up in `jobs.last_error`, so it is the only account of what went wrong."""
    done = subprocess.run(args, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{args[0]} failed: {done.stderr.strip()[-500:]}")
    return done.stdout


def _ffmpeg(*args):
    # `-protocol_whitelist file`: whatever the input turns out to be, it may not reach the network.
    # Belt to the braces of `_refuse_playlists` — that one stops the demuxers that go looking, this
    # one stops where they could look.
    _run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-protocol_whitelist", "file", *args)


def _ffprobe(entry, path):
    """One `format=` field as ffprobe reports it, stripped. Empty when it does not say."""
    return _run(
        "ffprobe", "-v", "error", "-protocol_whitelist", "file",
        "-show_entries", f"format={entry}",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ).strip()


def _duration(path):
    """Seconds as the container reports them, or None when it does not say."""
    try:
        return float(_ffprobe("duration", path))
    except ValueError:
        return None


def _refuse_playlists(path):
    """Stop before encoding anything if the «media» is really a list of other files to fetch.

    Raising here fails the job, which after three tries fails the item — the gallery then shows the
    placeholder it shows for anything that could not be prepared.
    """
    # ffprobe answers with a comma-separated list of every name the container goes by.
    named = {name for name in _ffprobe("format_name", path).split(",") if name}
    playlists = named & PLAYLIST_FORMATS
    if playlists:
        raise RuntimeError(f"refusing {'/'.join(sorted(playlists))}: a playlist is not media")


# --- the handler ------------------------------------------------------------


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

    s3 = storage()
    thumbnail_key, preview_key, duration = f"thumbnails/{ref}.jpg", None, None
    with tempfile.TemporaryDirectory() as workdir:
        workdir = pathlib.Path(workdir)
        original, thumbnail = workdir / "original", workdir / "thumbnail.jpg"
        s3.download_file(bucket(), original_key, str(original))
        # Before ffmpeg is pointed at it for anything else: these bytes came off somebody's phone
        # and nothing between there and here has looked inside them.
        _refuse_playlists(original)

        preview = workdir / ("preview.mp4" if kind == "video" else "preview.jpg")
        if kind == "video":
            duration = _duration(original)
            # A frame a little way in: the first one of a phone video is regularly black, and half
            # way is the fallback for a clip shorter than a second.
            at = min(1.0, duration / 2) if duration else 0
            _ffmpeg("-ss", str(at), "-i", str(original), "-frames:v", "1",
                    "-vf", THUMBNAIL_SCALE, str(thumbnail))
            _ffmpeg(
                "-i", str(original), "-vf", PREVIEW_SCALE,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                # The gallery plays this progressively (architecture.md): the index has to sit at
                # the front of the file or the player has to fetch all of it before the first frame.
                "-movflags", "+faststart", str(preview),
            )
            preview_key = f"previews/{ref}.mp4"
            _upload(s3, preview, preview_key, "video/mp4")
        else:
            _ffmpeg("-i", str(original), "-frames:v", "1", "-vf", THUMBNAIL_SCALE, str(thumbnail))
            # The full-screen copy. The original stays where it is — it is what the montage is cut
            # from — but a phone must not have to pull forty megapixels down to look at one frame.
            # `-q:v 3` because the whole point of this file is that it is sharp.
            _ffmpeg("-i", str(original), "-frames:v", "1", "-vf", PHOTO_PREVIEW_SCALE,
                    "-q:v", "3", str(preview))
            preview_key = f"previews/{ref}.jpg"
            _upload(s3, preview, preview_key, "image/jpeg")

        _upload(s3, thumbnail, thumbnail_key, "image/jpeg")

    # The measured length replaces the one the phone declared at intent, which was only ever a claim.
    # ponytail: `width`/`height` stay untouched — ffprobe reports the stored dimensions and ffmpeg
    # previews the EXIF-rotated ones, so the two disagree on every sideways phone photo. Fill them
    # when the gallery needs the aspect ratio before the thumbnail loads.
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
                s3.delete_object(Bucket=bucket(), Key=key)
        return
    log.info("media item %s (%s) is ready", ref, kind)


def _upload(s3, path, key, content_type):
    s3.upload_file(str(path), bucket(), key, ExtraArgs={"ContentType": content_type})


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


# --- the sweep --------------------------------------------------------------


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

    s3, swept = storage(), 0
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
            if upload_id:
                try:
                    s3.abort_multipart_upload(
                        Bucket=bucket(), Key=original_key, UploadId=upload_id
                    )
                except s3.exceptions.NoSuchUpload:
                    pass  # already gone; the row has followed it either way
            # And whatever landed under the key anyway, for the client that completed its upload in
            # R2 and never told the api.
            s3.delete_object(Bucket=bucket(), Key=original_key)
        except Exception as exc:
            # The row is already gone, so nothing will ever name these again and the next hour will
            # not find them. The log is the only record left — hence the upload id in it.
            log.warning("swept %s but could not free it in R2 (upload %s), left behind: %s",
                        original_key, upload_id, exc)
        swept += 1

    log.info("swept %s abandoned upload(s) of %s found", swept, len(stale))
    return swept
