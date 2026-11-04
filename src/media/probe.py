import json
import logging
import math
import pathlib
import re
import subprocess
import sys
from datetime import datetime
from fractions import Fraction

from src.integrations import ffmpeg

log = logging.getLogger("media.probe")

TIMEOUT_SECONDS = 60
UPENDED = {5, 6, 7, 8}
ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _shutter(value):
    try:
        return str(Fraction(value.numerator, value.denominator))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return None


def _degrees(value, hemisphere):
    parts = [_number(part) for part in value] if isinstance(value, tuple) else []
    if len(parts) != 3 or None in parts:
        return None
    degrees = parts[0] + parts[1] / 60 + parts[2] / 3600
    return -degrees if str(hemisphere or "").strip().upper() in ("S", "W") else degrees


def _read_photo(path: str) -> dict:
    import pi_heif
    from PIL import ExifTags, Image

    pi_heif.register_heif_opener()
    with Image.open(path) as photo:
        width, height = photo.size
        exif = photo.getexif()
        if exif.get(ExifTags.Base.Orientation) in UPENDED:
            width, height = height, width
        model = exif.get(ExifTags.Base.Model)
        optics = exif.get_ifd(ExifTags.IFD.Exif)
        place = exif.get_ifd(ExifTags.IFD.GPSInfo)
        return {
            "width": width,
            "height": height,
            "duration": None,
            "captured_at": _exif_time(optics.get(ExifTags.Base.DateTimeOriginal)),
            "camera_model": str(model or "").strip() or None,
            "focal_length_mm": _number(optics.get(ExifTags.Base.FocalLength)),
            "aperture": _number(optics.get(ExifTags.Base.FNumber)),
            "iso": _number(optics.get(ExifTags.Base.ISOSpeedRatings)),
            "exposure_ev": _number(optics.get(ExifTags.Base.ExposureBiasValue)),
            "shutter": _shutter(optics.get(ExifTags.Base.ExposureTime)),
            "latitude": _degrees(
                place.get(ExifTags.GPS.GPSLatitude), place.get(ExifTags.GPS.GPSLatitudeRef)
            ),
            "longitude": _degrees(
                place.get(ExifTags.GPS.GPSLongitude), place.get(ExifTags.GPS.GPSLongitudeRef)
            ),
        }


def _exif_time(value) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return None


def _clip(path: pathlib.Path, kind: str) -> dict:
    data = ffmpeg.probe(path)
    tags = data.get("format", {}).get("tags", {})
    width = height = None
    if kind == "video":
        found = ffmpeg.dimensions(path)
        if found:
            width, height = found
    place = tags.get("com.apple.quicktime.location.ISO6709") or tags.get("location")
    where = ISO6709.match(place.strip()) if place else None
    return {
        "width": width,
        "height": height,
        "duration": ffmpeg.duration(path),
        "captured_at": tags.get("creation_time"),
        "camera_model": tags.get("com.apple.quicktime.model") or tags.get("model") or None,
        "focal_length_mm": None,
        "aperture": None,
        "iso": None,
        "exposure_ev": None,
        "shutter": None,
        "latitude": float(where.group(1)) if where else None,
        "longitude": float(where.group(2)) if where else None,
    }


def _plausible(info: dict) -> dict:
    def positive(value):
        return value if value is not None and math.isfinite(value) and value > 0 else None

    def stated(value):
        return value if value is not None and math.isfinite(value) else None

    latitude, longitude = info.get("latitude"), info.get("longitude")
    place = (
        latitude is not None
        and longitude is not None
        and abs(latitude) <= 90
        and abs(longitude) <= 180
        and (latitude, longitude) != (0, 0)
    )
    model = "".join(char for char in (info.get("camera_model") or "") if char.isprintable())
    return {
        **info,
        "camera_model": model.strip()[:100] or None,
        "focal_length_mm": positive(info.get("focal_length_mm")),
        "aperture": positive(info.get("aperture")),
        "iso": positive(info.get("iso")),
        "exposure_ev": stated(info.get("exposure_ev")),
        "latitude": latitude if place else None,
        "longitude": longitude if place else None,
    }


def _ask_photo(path: pathlib.Path) -> dict | None:
    try:
        done = subprocess.run(
            [sys.executable, "-m", "src.media.probe", str(path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        log.warning("photo probe timed out after %ss", TIMEOUT_SECONDS)
        return None
    if done.returncode != 0:
        log.warning("photo probe failed: %s", done.stderr.strip().splitlines()[-1:])
        return None
    return json.loads(done.stdout or "null")


def read(path: pathlib.Path, kind: str) -> dict:
    if kind == "photo":
        info = _ask_photo(path)
    else:
        info = _clip(path, kind)
    return _plausible(info or {})


def _render(path: str, out: str, side: str) -> dict:
    import pi_heif
    from PIL import Image, ImageOps

    pi_heif.register_heif_opener()
    with Image.open(path) as photo:
        upright = ImageOps.exif_transpose(photo).convert("RGB")
        upright.thumbnail((int(side), int(side)))
        upright.save(out, "JPEG", quality=85)
        return {"width": upright.width, "height": upright.height}


def render_photo(path: pathlib.Path, out: pathlib.Path, side: int) -> dict:
    done = subprocess.run(
        [sys.executable, "-m", "src.media.probe", "render", str(path), str(out), str(side)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )
    if done.returncode != 0:
        raise RuntimeError(f"photo render failed: {' '.join(done.stderr.splitlines()[-2:])}")
    return json.loads(done.stdout)


def main() -> None:
    if sys.argv[1] == "render":
        print(json.dumps(_render(*sys.argv[2:])))
        return
    print(json.dumps(_read_photo(sys.argv[1])))


if __name__ == "__main__":
    main()
