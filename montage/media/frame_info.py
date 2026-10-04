"""The `media_frame_info` job: how big a frame is and what its camera wrote into it.

The lines «Размер», «Камера» and «Место» of the sheet «Сведения о кадре» (doc-02, the layout in
`design/screens/SheetFrameInfo.html`), and the strip of shooting settings beside them — ISO, focal
length, exposure, aperture, shutter — read out of the file the phone uploaded and written onto the
row for the api to hand out — exactly as `thumbnail_key` is (the api's migrations
`100.do.frame-info.sql` and `200.do.frame-exposure.sql`). Nothing here is computed: what the
camera did not write down stays null.

A job of its own rather than one more step of the preview, so that the original is read once for
every answer the sheet wants off the file, and the catch-up that fills them in for old material
costs a download and a probe — never the whole re-encode a second preview would be.
"""

import collections
import logging
import math
import pathlib
import re
import tempfile

from montage import ffmpeg, photo, storage

log = logging.getLogger("montage.media.frame_info")


FrameInfo = collections.namedtuple(
    "FrameInfo",
    "width height camera_model focal_length_mm aperture iso exposure_ev shutter latitude longitude",
)
NOTHING = FrameInfo(*(None,) * 10)  # a file that says nothing at all
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
    tags = ffmpeg.format_tags(path)
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

    def stated(value):
        """Whether the camera said so at all — for a number that may legitimately be zero. The
        exposure compensation is such a number: `0` is «shot without compensation» and means the
        same thing to a reader as any other reading, so it is kept and told apart from the null a
        camera that never wrote it leaves behind."""
        return value if value is not None and math.isfinite(value) else None

    model = "".join(char for char in info.camera_model or "" if char.isprintable())
    a_place = (
        info.latitude is not None
        and info.longitude is not None
        and abs(info.latitude) <= 90
        and abs(info.longitude) <= 180
        and (info.latitude, info.longitude) != (0, 0)
    )
    return info._replace(
        camera_model=model.strip()[:MODEL_LENGTH] or None,
        focal_length_mm=positive(info.focal_length_mm),
        aperture=positive(info.aperture),
        # Sensitivity is a length of the same kind: there is no ISO below one.
        iso=positive(info.iso),
        exposure_ev=stated(info.exposure_ev),
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
        width, height = ffmpeg.dimensions(path) or (None, None)
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
        storage.download(original_key, original)
        # These bytes came off somebody's phone and ffprobe is about to be pointed at them: the
        # same guard the preview puts in front of ffmpeg, for the same reason.
        ffmpeg.refuse_playlists(original)
        info = frame_info(original, kind)

    conn.execute(
        """UPDATE media_items
           SET width = %s, height = %s, camera_model = %s, focal_length_mm = %s, aperture = %s,
               iso = %s, exposure_ev = %s, shutter = %s,
               latitude = %s, longitude = %s, updated_at = now()
           WHERE id = %s""",
        (*info, ref),
    )
    # Which of them, not what: the place is where somebody was, and a log is read by more people
    # (and kept longer) than the sheet is shown to.
    found = [name for name, value in info._asdict().items() if value is not None]
    log.info(
        "media item %s: read %s", ref, ", ".join(found) or "nothing about size, camera or place"
    )
