"""The `media_preview` and `media_frame_info` jobs, and the hourly sweep of uploads that were never
finished.

The api writes a `media_items` row, takes the bytes into R2 and queues the job; from `processing`
onward the row is this worker's (see the api's migration `004.do.media.sql`). Here the original is
pulled back out, `ffmpeg` makes what the gallery actually shows — a thumbnail for the grid and, for
both kinds, a lighter full-screen copy (2048px JPEG for a photo, 720p mp4 for a video) — and the row
is left `ready` with the derivative keys on it.

`media_items` belongs to the api's migrations. This module only reads and writes rows by id.
"""

import collections
import json
import logging
import math
import os
import pathlib
import re
import subprocess
import tempfile

import boto3
from botocore.config import Config

import photo

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
PROBE_SECONDS = 60  # ffprobe reads a header; one still at it a minute on is not coming back
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


def _run(*args, timeout=None):
    """A tool, its output, and its stderr in the exception when it fails — that text is what ends
    up in `jobs.last_error`, so it is the only account of what went wrong."""
    done = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
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
        timeout=PROBE_SECONDS,
    ).strip()


def _duration(path):
    """Seconds as the container reports them, or None when it does not say."""
    try:
        return float(_ffprobe("duration", path))
    except ValueError:
        return None


def _dimensions(path):
    """`(width, height)` of the picture as it is *seen*, or None when ffprobe does not say.

    A clip the phone was held upright for is stored on its side with a display matrix beside it
    saying to stand it up, and answers here the way it plays — taller than wide. Everything that
    decodes it sees the same picture: ffmpeg turns it on the way out unless told not to.

    A photograph's turn is written in EXIF instead, which ffprobe does not hand over as a
    rotation — those are measured by Pillow (`photo.read`).
    """
    probed = _run(
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:stream_side_data=rotation", "-of", "json", str(path),
        timeout=PROBE_SECONDS,
    )
    try:
        stream, = json.loads(probed)["streams"]
        width, height = stream["width"], stream["height"]
    except (ValueError, KeyError):
        # Nothing to read the size off: no picture in the file at all, or more than one answer.
        # Callers treat that as «cannot say» and fall back on what they had.
        return None
    if width < 1 or height < 1:
        # A header ffmpeg will not believe — a size it would refuse to allocate — comes back as
        # zeros, and «0 × 0» on the sheet is no better an answer than none.
        return None
    # A quarter turn either way swaps the sides; half a turn and none leave them as they are.
    upended = any(abs(side.get("rotation", 0)) % 180 == 90
                  for side in stream.get("side_data_list", ()))
    return (height, width) if upended else (width, height)


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
            # A HEIC goes to ffmpeg already stood up: it does not apply the file's own turn
            # (`photo.stand_up`), and JPEG's EXIF orientation it does.
            upright = workdir / "upright.jpg"
            seen = upright if photo.stand_up(original, upright) else original
            _ffmpeg("-i", str(seen), "-frames:v", "1", "-vf", THUMBNAIL_SCALE, str(thumbnail))
            # The full-screen copy. The original stays where it is — it is what the montage is cut
            # from — but a phone must not have to pull forty megapixels down to look at one frame.
            # `-q:v 3` because the whole point of this file is that it is sharp.
            _ffmpeg("-i", str(seen), "-frames:v", "1", "-vf", PHOTO_PREVIEW_SCALE,
                    "-q:v", "3", str(preview))
            preview_key = f"previews/{ref}.jpg"
            _upload(s3, preview, preview_key, "image/jpeg")

        _upload(s3, thumbnail, thumbnail_key, "image/jpeg")

    # The measured length replaces the one the phone declared at intent, which was only ever a claim.
    # `width`/`height` are not written here but by `media_frame_info` below, which is queued at the
    # end of this handler: they are read off the original like the camera and the place, and one
    # owner for the lot means one answer and one catch-up.
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
    # Two jobs of the отсев's and speech's own, queued by hand rather than through
    # `worker.enqueue` — `worker` imports this module, and the import cannot go both ways.
    # The screening proposal is every kind's (`screening.py`): the organizer meets it in the
    # gallery long before any video is ordered. Speech is a clip's alone (`speech.py`) — a cloud
    # model listening to one takes minutes, which the gallery does not wait on.
    conn.execute("INSERT INTO jobs (type, ref) VALUES ('media_screening', %s)", (ref,))
    if kind == "video":
        conn.execute("INSERT INTO jobs (type, ref) VALUES ('media_speech', %s)", (ref,))

    # And what the camera itself wrote into the original — its model, the optics, the place. The
    # sheet «Сведения о кадре» shows all three long before any video is ordered, and reading them
    # is `media_frame_info` below.
    conn.execute("INSERT INTO jobs (type, ref) VALUES ('media_frame_info', %s)", (ref,))

    # And straight on to what is in it: the frame is describable the moment the copies exist, and
    # the director wants it described long before anybody orders a video (`understanding.py`). The
    # row is inserted here rather than through `worker.enqueue` because the worker imports this
    # module and not the other way round — the same one line the api writes to queue this very job.
    conn.execute("INSERT INTO jobs (type, ref) VALUES ('media_understanding', %s)", (ref,))


def _upload(s3, path, key, content_type):
    s3.upload_file(str(path), bucket(), key, ExtraArgs={"ContentType": content_type})


def large_copy(s3, preview_key, into):
    """A photograph's 2048px copy out of the bucket, or None when it will not come.

    The copy `media_preview` made above, fetched by everything that would rather look at a
    photograph than at the forty megapixels it was uploaded at: the отсев judges it
    (`screening.media_screening`, `montage.montage`) and the montage cuts its pieces from it.

    None and not an exception, because nobody who asks for it has to have it: the original is the
    file this copy was made from, and every caller can fall back on it — slower, and for the отсев
    measured on a coarser sample, but never wrong. A copy that is missing, not made yet, or in an
    unreachable bucket is not worth failing an ordered video or a screening proposal over.
    """
    try:
        s3.download_file(bucket(), preview_key, str(into))
        return into
    except Exception as exc:
        log.warning("could not take the large copy %s, falling back on the original: %s",
                    preview_key, exc)
        return None


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


# --- the size, the camera and the place --------------------------------------
# The lines «Размер», «Камера» and «Место» of the sheet «Сведения о кадре» (doc-02, the layout in
# `design/screens/SheetFrameInfo.html`), read out of the file the phone uploaded and written onto
# the row for the api to hand out — exactly as `thumbnail_key` is (the api's migration
# `100.do.frame-info.sql`). Nothing here is computed: what the camera did not write down stays null.

FrameInfo = collections.namedtuple(
    "FrameInfo", "width height camera_model focal_length_mm aperture latitude longitude"
)
NOTHING = FrameInfo(None, None, None, None, None, None, None)  # a file that says nothing at all
MODEL_LENGTH = 100  # «iPhone 15 Pro», «SM-S918B» — a camera's name, not a paragraph

# ±DD.DDDD±DDD.DDDD±AAA.AAA/ — how a phone writes down where a clip was shot (ISO 6709). The third
# number is the altitude, which nobody asks for.
# ponytail: decimal degrees only. The standard also allows ±DDMM.MM, which reads here as degrees far
# off the globe and is dropped by `_plausible` — a camera that writes it lands as no place at all;
# add the minutes when one turns up.
ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


def _photo_info(path):
    """A photograph's size and its own EXIF — HEIC off an iPhone as readily as JPEG.

    Read by a library and not by ffprobe, which is otherwise this module's every answer about a
    file: ffprobe hands EXIF over only as frame tags of a JPEG and out of HEIC not at all, so
    whether the sheet had a camera line would depend on the file's format and the host's ffmpeg.
    The library runs in a process of its own (`photo.py`): a corrupt HEIC can crash it.
    """
    found = photo.read(path)
    # Not a picture the library knows, or the reader died — `kind` is only the phone's word for the
    # file. Nothing to read is not a failure: the frame keeps its empty columns like any other, and
    # the size is asked of ffprobe instead (`frame_info`).
    return FrameInfo(**found) if found else NOTHING


def _clip_info(path):
    """A clip's own tags, off the container: the camera and the place, which a phone writes there,
    and nothing about the optics, which it writes nowhere. Nothing is decoded for these."""
    tags = json.loads(_run(
        "ffprobe", "-v", "error", "-protocol_whitelist", "file",
        "-show_entries", "format_tags", "-of", "json", str(path), timeout=PROBE_SECONDS,
    )).get("format", {}).get("tags", {})
    # An iPhone keeps both under its own names; ffmpeg gives Android's place as a plain `location`.
    place = tags.get("com.apple.quicktime.location.ISO6709") or tags.get("location")
    where = ISO6709.match(place.strip()) if place else None
    return NOTHING._replace(
        camera_model=(tags.get("com.apple.quicktime.model") or tags.get("model") or "").strip()
                     or None,
        latitude=float(where[1]) if where else None,
        longitude=float(where[2]) if where else None,
    )


def _plausible(info):
    """What the camera wrote, minus what no column and no map can use. Numbers: NaN and infinity (an
    EXIF rational over zero, a run of digits in a clip's place tag), an optic that is not a positive
    length, a place off the globe — a tag in degrees-and-minutes read as decimal degrees — and the
    point (0, 0) itself, which is where a camera with no satellite fix writes its place. And the
    model, which is the file's own text served to every member of the memory: one short line, since
    a NUL in it is refused by the `text` column, which fails the job three times over for good."""
    def positive(value):
        return value if value is not None and math.isfinite(value) and value > 0 else None

    model = "".join(char for char in info.camera_model or "" if char.isprintable())
    a_place = (info.latitude is not None and info.longitude is not None
               and abs(info.latitude) <= 90 and abs(info.longitude) <= 180
               and (info.latitude, info.longitude) != (0, 0))
    return info._replace(
        camera_model=model.strip()[:MODEL_LENGTH] or None,
        focal_length_mm=positive(info.focal_length_mm),
        aperture=positive(info.aperture),
        latitude=info.latitude if a_place else None,
        longitude=info.longitude if a_place else None,
    )


def frame_info(path, kind):
    """How big the frame is, and what the camera left in the file — None in place of everything it
    left out. Most material carries none of the latter: a screenshot, a frame re-saved by an
    editor, anything that came through a messenger.

    The size is the one line almost every frame can answer, and it is measured the way the frame is
    *seen*: a photograph taken sideways and a clip shot upright both come back taller than wide.
    A file with no picture in it at all keeps those two columns empty like the rest.

    The place is the coordinates and stays the coordinates. Turning them into «Усадьба Остафьево»
    takes an outside geocoder — a paid service, a registration, and a trip to the network with
    somebody's whereabouts in the query; whether the product wants a name at all is the chief's
    question and not one to guess at here (TASK-191).
    """
    info = _clip_info(path) if kind == "video" else _photo_info(path)
    if info.width is None:
        # Every clip, and a «photograph» Pillow would not open — `kind` is only the phone's word for
        # the file. ffprobe measures what the preview was made out of, so anything that reached the
        # gallery at all has a size here.
        width, height = _dimensions(path) or (None, None)
        info = info._replace(width=width, height=height)
    return _plausible(info)


def media_frame_info(conn, ref):
    """One media item: the original out of R2, how big its picture is and what its camera wrote
    down, onto the row.

    A job of its own rather than one more step of `media_preview`, so that the original is read
    once for every answer the sheet wants off the file, and the catch-up that fills them in for old
    material costs a download and a probe — never the whole re-encode a second preview would be.
    """
    row = conn.execute(
        "SELECT kind, original_key FROM media_items WHERE id = %s", (ref,)
    ).fetchone()
    if row is None:
        # Deleted while the job waited — a member left and took their material with them. Nothing
        # to read and nothing wrong: retrying would only fail three times.
        log.info("media item %s is gone, nothing to read a camera off", ref)
        return
    kind, original_key = row

    with tempfile.TemporaryDirectory() as workdir:
        original = pathlib.Path(workdir) / "original"
        storage().download_file(bucket(), original_key, str(original))
        # These bytes came off somebody's phone and ffprobe is about to be pointed at them: the
        # same guard the preview puts in front of ffmpeg, for the same reason.
        _refuse_playlists(original)
        info = frame_info(original, kind)

    conn.execute(
        """UPDATE media_items
           SET width = %s, height = %s, camera_model = %s, focal_length_mm = %s, aperture = %s,
               latitude = %s, longitude = %s, updated_at = now()
           WHERE id = %s""",
        (*info, ref),
    )
    # Which of them, not what: the place is where somebody was, and a log is read by more people
    # (and kept longer) than the sheet is shown to.
    found = [name for name, value in info._asdict().items() if value is not None]
    log.info("media item %s: read %s", ref, ", ".join(found) or "nothing about size, camera or place")


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
