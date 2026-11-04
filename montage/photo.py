import json
import logging
import subprocess
import sys
from fractions import Fraction

log = logging.getLogger("montage.photo")

TIMEOUT_SECONDS = 60

FORMATS = ("JPEG", "HEIF")

UPENDED = {5, 6, 7, 8}


def read(path):
    return _ask("read", path)


def stand_up(path, out):
    try:
        with open(path, "rb") as file:
            if file.read(12)[4:8] != b"ftyp":
                return False
    except OSError:
        return False
    return _ask("stand-up", path, out) is True


def _ask(*args):
    try:
        done = subprocess.run(
            [sys.executable, __file__, *map(str, args)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "the photo reader ran past %ss, taking it for a file with nothing to read",
            TIMEOUT_SECONDS,
        )
        return None
    if done.returncode != 0:
        log.warning(
            "the photo reader died (exit %s), taking it for a file with nothing to read",
            done.returncode,
        )
        return None
    if done.stderr.strip():
        log.info("the photo reader: %s", done.stderr.strip().splitlines()[-1])
    return json.loads(done.stdout)


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


def _open(path, formats):
    import pi_heif
    from PIL import Image

    pi_heif.register_heif_opener()
    return Image.open(path, formats=formats)


def _read(path):
    from PIL import ExifTags

    with _open(path, FORMATS) as photo:
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


def _stand_up(path, out):
    with _open(path, ("HEIF",)) as photo:
        photo.convert("RGB").save(out, "JPEG", quality=95)
    return True


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    try:
        answer = {"read": _read, "stand-up": _stand_up}[command](*arguments)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        answer = None
    print(json.dumps(answer))
