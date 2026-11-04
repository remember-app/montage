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
NOTHING = FrameInfo(*(None,) * 10)
MODEL_LENGTH = 100

ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


def _photo_info(path):
    found = photo.read(path)
    return FrameInfo(**found) if found else NOTHING


def _clip_info(path):
    tags = ffmpeg.format_tags(path)
    place = tags.get("com.apple.quicktime.location.ISO6709") or tags.get("location")
    where = ISO6709.match(place.strip()) if place else None
    return NOTHING._replace(
        camera_model=(tags.get("com.apple.quicktime.model") or tags.get("model") or "").strip()
        or None,
        latitude=float(where[1]) if where else None,
        longitude=float(where[2]) if where else None,
    )


def _plausible(info):

    def positive(value):
        return value if value is not None and math.isfinite(value) and value > 0 else None

    def stated(value):
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
        iso=positive(info.iso),
        exposure_ev=stated(info.exposure_ev),
        latitude=info.latitude if a_place else None,
        longitude=info.longitude if a_place else None,
    )


def frame_info(path, kind):
    info = _clip_info(path) if kind == "video" else _photo_info(path)
    if info.width is None:
        width, height = ffmpeg.dimensions(path) or (None, None)
        info = info._replace(width=width, height=height)
    return _plausible(info)


def media_frame_info(conn, ref):
    row = conn.execute(
        "SELECT kind, original_key FROM media_items WHERE id = %s", (ref,)
    ).fetchone()
    if row is None:
        log.info("media item %s is gone, nothing to read a camera off", ref)
        return
    kind, original_key = row

    with tempfile.TemporaryDirectory() as workdir:
        original = pathlib.Path(workdir) / "original"
        storage.download(original_key, original)
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
    found = [name for name, value in info._asdict().items() if value is not None]
    log.info(
        "media item %s: read %s", ref, ", ".join(found) or "nothing about size, camera or place"
    )
