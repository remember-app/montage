"""What Pillow and libheif know about a photograph, asked in a process of its own.

Both are C code and the bytes they are handed came off somebody's phone — or off somebody who wanted
the worker to fall over. A HEIC with a few bytes flipped, one ffmpeg still cuts a thumbnail out of,
segfaults libheif inside `Image.open` (three of a thousand flipped fixtures did), and a segfault in
the worker's own process is the worker gone with the job it held. So the asking is a child that may
die: whatever it printed is the answer, and a child that crashed, hung or ran out of memory is a
file with nothing to read. That contains a crash, it is not a sandbox: the child runs as the worker.

Run as a script this file is the child. The parent side is `read` and `stand_up`, and it imports
nothing but the standard library, so the worker never loads libheif into itself.
"""

import json
import logging
import subprocess
import sys

log = logging.getLogger("montage.photo")

TIMEOUT_SECONDS = 60  # a HEIC of many megapixels takes seconds to decode; a hang takes forever

# What a phone's camera writes, and what Pillow reads the EXIF of without decoding a picture (a JPEG
# with several frames is opened as MPO by the JPEG plugin itself). Everything else is measured by
# ffprobe and has no camera line. PNG above all: its `getexif()` decodes the whole image — a 2 MB,
# 169-megapixel file took a second and 660 MB — and no camera writes one.
FORMATS = ("JPEG", "HEIF")

# The EXIF orientations that lay the picture on its side: the sensor wrote its frame one way round
# and the person sees it the other. «6» is the sideways phone photograph every gallery is full of.
UPENDED = {5, 6, 7, 8}


# --- the parent -------------------------------------------------------------


def read(path):
    """The size a person sees and what the camera wrote — `width`, `height`, `camera_model`,
    `focal_length_mm`, `aperture`, `latitude`, `longitude`, each None when the file does not say.
    None instead of the lot when this is not a photograph the reader opens, or the reader died."""
    return _ask("read", path)


def stand_up(path, out):
    """A HEIC turned the way it is seen, as a JPEG at `out`; False when `path` is not one, or could
    not be decoded, and the caller goes on with the original.

    ffmpeg (7.1) leaves the `irot` box of a HEIC unapplied, so a portrait photograph off an iPhone —
    every one of them stores its sensor's frame and a note to turn it — came out of it lying on its
    side. libheif applies it while it decodes. The JPEG carries no orientation of its own, so nothing
    downstream can turn it twice.
    """
    with open(path, "rb") as file:
        # ftyp is how every ISO base media file starts, HEIC and mp4 alike; the child says which.
        if file.read(12)[4:8] != b"ftyp":
            return False
    return _ask("stand-up", path, out) is True


def _ask(*args):
    try:
        done = subprocess.run([sys.executable, __file__, *map(str, args)],
                              capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        log.warning("the photo reader ran past %ss, taking it for a file with nothing to read",
                    TIMEOUT_SECONDS)
        return None
    if done.returncode != 0:
        # A signal comes back as a negative number: -11 is libheif or libjpeg falling over.
        log.warning("the photo reader died (exit %s), taking it for a file with nothing to read",
                    done.returncode)
        return None
    if done.stderr.strip():
        log.info("the photo reader: %s", done.stderr.strip().splitlines()[-1])
    return json.loads(done.stdout)


# --- the child --------------------------------------------------------------


def _number(value):
    """A number as EXIF holds it — a rational, as a rule. None when the tag is not there at all,
    and when what is in it will not become a number."""
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _degrees(value, hemisphere):
    """One EXIF coordinate: degrees, minutes and seconds in one tag, north or south in another."""
    parts = [_number(part) for part in value] if isinstance(value, tuple) else []
    if len(parts) != 3 or None in parts:
        return None
    degrees = parts[0] + parts[1] / 60 + parts[2] / 3600
    return -degrees if str(hemisphere or "").strip().upper() in ("S", "W") else degrees


def _open(path, formats):
    import pi_heif
    from PIL import Image

    pi_heif.register_heif_opener()  # HEIC is what an iPhone uploads; Pillow opens it once told
    return Image.open(path, formats=formats)


def _read(path):
    from PIL import ExifTags

    with _open(path, FORMATS) as photo:
        width, height = photo.size
        exif = photo.getexif()
        if exif.get(ExifTags.Base.Orientation) in UPENDED:
            # The sideways phone: the file holds the sensor's frame and a note saying which way up it
            # goes, and the size a person sees is that note applied. HEIC comes out of `pi_heif`
            # already turned, with the note reset to «1», so this is JPEG's case.
            width, height = height, width
        model = exif.get(ExifTags.Base.Model)
        optics = exif.get_ifd(ExifTags.IFD.Exif)
        place = exif.get_ifd(ExifTags.IFD.GPSInfo)
        return {
            "width": width,
            "height": height,
            # EXIF keeps the make («Apple») in a tag of its own — «iPhone 15 Pro» is the model alone.
            "camera_model": str(model or "").strip() or None,
            "focal_length_mm": _number(optics.get(ExifTags.Base.FocalLength)),
            # The f-number, so that 1.8 is what the app puts on screen as «f/1.8».
            "aperture": _number(optics.get(ExifTags.Base.FNumber)),
            "latitude": _degrees(place.get(ExifTags.GPS.GPSLatitude),
                                 place.get(ExifTags.GPS.GPSLatitudeRef)),
            "longitude": _degrees(place.get(ExifTags.GPS.GPSLongitude),
                                  place.get(ExifTags.GPS.GPSLongitudeRef)),
        }


def _stand_up(path, out):
    with _open(path, ("HEIF",)) as photo:
        photo.convert("RGB").save(out, "JPEG", quality=95)
    return True


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    try:
        answer = {"read": _read, "stand-up": _stand_up}[command](*arguments)
    except Exception as exc:  # whatever it was, the file has nothing to read
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        answer = None
    print(json.dumps(answer))
